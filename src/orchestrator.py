from dataset import DatasetDownloader
from embeddings import EmbeddingService
from llm import LLMService
from clustering import cluster_with_hdbscan, cluster_with_kmeans, get_representative_examples, calculate_all_davies_bouldin
from sklearn.preprocessing import normalize
from agent_actions import name_cluster, parse_feedback
import numpy as np
from config import ROOT_DIR
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

from cluster_state import (
    rename_cluster_state,
    binary_merge_cluster_state,
    split_cluster_state,
    move_item_cluster_state,
)

def handle_needs_clarification(arguments: dict) -> dict:
    question = arguments["question"]
    reason = arguments["reason"]

    return {
        "status": "needs_clarification",
        "question": question,
        "reason": reason,
    }

def handle_no_action(arguments: dict) -> dict:
    reason = arguments["reason"]

    return {
        "status": "no_action",
        "reason": reason,
    }
    
    
ACTION_REGISTRY = {
    "rename_cluster": lambda self, args: rename_cluster_state(
        llm_service=self.llm_service,
        data=self.data,
        cluster_metadata=self.cluster_metadata,
        cluster_id=args["cluster_id"],
        reason=args["reason"],
    ),

    "merge_clusters": lambda self, args: binary_merge_cluster_state(
        llm_service=self.llm_service,
        data=self.data,
        cluster_metadata=self.cluster_metadata,
        cluster_id_1=args["cluster_id_1"],
        cluster_id_2=args["cluster_id_2"],
    ),

    "split_cluster": lambda self, args: split_cluster_state(
        data=self.data,
        cluster_metadata=self.cluster_metadata,
        cluster_id=args["cluster_id"],
        llm_service=self.llm_service,
    ),

    "move_item": lambda self, args: move_item_cluster_state(
        data=self.data,
        item_id=args["item_id"],
        target_cluster_id=args["target_cluster_id"],
    ),
    "needs_clarification": lambda self, args: handle_needs_clarification(args),
    "no_action": lambda self, args: handle_no_action(args),
}


class Orchestrator:
    def __init__(
        self,
        sample_size: int | None = 100,
        random_state: int = 42,
    ) -> None:
        self.sample_size = sample_size
        self.random_state = random_state

        self.dataset = DatasetDownloader()
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()

        self.data = None
        self.embeddings = None
        self.labels = None
        self.cluster_metadata = {}
    
    def load_dataset(self) -> None:
        console.log("Caricamento dataset...")
        self.dataset.download_data()
        self.data = self.dataset.load_data(
            sample_size=self.sample_size,
            random_state=self.random_state
        )
        console.log(f"Dataset caricato — {len(self.data):,} righe")

    def generate_embeddings(self) -> None:
        console.log("Generazione embeddings...")
        cache_path = ROOT_DIR / "data" / "embeddings.npz"
        self.data = self.embedding_service.attach_embeddings_to_dataframe(
            df=self.data,
            cache_path=cache_path,
        )
        console.log("Embeddings pronti")

    def cluster_data(self, method: str = "hdbscan", k: int | None = None) -> None:
        console.log(f"Clustering con {method}{f' (k={k})' if k else ''}...")
        self.embeddings = np.vstack(self.data["embedding"].to_numpy())
        self.embeddings = normalize(self.embeddings)

        if method == "hdbscan":
            self.labels = cluster_with_hdbscan(self.embeddings)
        elif method == "kmeans":
            if k is None:
                raise ValueError("k must be provided for kmeans clustering")
            self.labels = cluster_with_kmeans(self.embeddings, k=k)
        else:
            raise ValueError("Invalid clustering type. Use 'hdbscan' or 'kmeans'.")

        self.data["cluster_id"] = self.labels
        n_clusters = len(set(self.labels) - {-1})
        console.log(f"Trovati {n_clusters} cluster")

    def label_clusters(self) -> None:
        unique_cluster_ids = self.data["cluster_id"].unique()
        console.log(f"Etichettatura {len(unique_cluster_ids)} cluster...")

        for cluster_id in unique_cluster_ids:
            if cluster_id == -1:
                self.cluster_metadata[cluster_id] = {
                    "name": "Noise / Outliers",
                    "description": "Items not assigned to a stable cluster.",
                    "evidence": [],
                }
                continue

            console.log(f"   → Cluster {cluster_id}...")
            examples = get_representative_examples(data=self.data, cluster_id=cluster_id)
            result = name_cluster(llm_service=self.llm_service, examples=examples)
            self.cluster_metadata[cluster_id] = result

        console.log("Cluster etichettati")
    
    def apply_feedback(self, user_feedback: str):
        parsed_action = parse_feedback(
            llm_service=self.llm_service,
            user_feedback=user_feedback,
            cluster_metadata=self.cluster_metadata,
        )

        action_name = parsed_action.get("tool_name")
        arguments = parsed_action.get("arguments", {})

        if action_name not in ACTION_REGISTRY:
            raise ValueError(f"Unknown action: {action_name}")

        action_function = ACTION_REGISTRY[action_name]
        return action_function(self, arguments)

    def show_clusters(self) -> None:
        console.rule("[bold blue]Current clusters")
        davies_bouldin = calculate_all_davies_bouldin(self.data)

        for cluster_id, meta in sorted(self.cluster_metadata.items()):
            count = int((self.data["cluster_id"] == cluster_id).sum())
            
            examples = get_representative_examples(
                data=self.data,
                cluster_id=cluster_id,
                samples_per_tercile=2,
                seed=42,
            )

            header = Text()
            header.append(f"Cluster {cluster_id}", style="bold cyan")
            header.append(" · ")
            header.append(meta["name"], style="bold white")

            info_table = Table.grid(padding=(0, 2))
            info_table.add_column(style="magenta", justify="right")
            info_table.add_column(style="white")

            info_table.add_row("Description", meta["description"])
            info_table.add_row("Items", str(count))
            info_table.add_row("Davies-Bouldin", f"{davies_bouldin.get(cluster_id, 'N/A'):.3f}")

            examples_table = Table(
                title="Representative examples",
                show_header=True,
                header_style="bold green"
            )
            examples_table.add_column("Id", style="yellow", width=8)
            examples_table.add_column("Text", style="white")

            for ex in examples:
                text = ex["Text"].replace("\n", " ").strip()
                examples_table.add_row(str(ex["Id"]), text)

            console.print(Panel.fit(header, border_style="blue"))
            console.print(info_table)
            console.print(examples_table)
            console.print()

    def run(self) -> None:
        # 1. Carica il dataset
        self.load_dataset()
        # 2. Genera gli embedding e li salva nel DataFrame
        self.generate_embeddings()
        # 3. Genero i cluster iniziali
        self.cluster_data(method="kmeans", k=5)
        
        # 4. Nomino i cluster inziali
        self.label_clusters()
        self.show_clusters()

        # 5. Loop con feedback umano
        while True:
            user_input = input(
                "\nProvide feedback, or type 'quit', 'exit', 'stop', or 'q' to end: "
            ).strip()

            if user_input.lower() in {"quit", "exit", "stop", "q"}:
                print("Exiting interactive loop.")
                break

            if not user_input:
                print("Empty input. Please write feedback or type 'quit'.")
                continue

            try:
                result = self.apply_feedback(user_input)

                if result.get("status") == "needs_clarification":
                    print("\nClarification needed:")
                    print(result["question"])
                    print(f"Reason: {result['reason']}")
                    continue
                if result.get("status") == "no_action":
                    print("\nNo action taken:")
                    print(f"Reason: {result['reason']}")
                    continue

                print("\nApplied action:")
                print(result)

            except Exception as e:
                print(f"\nCould not apply feedback: {e}")
                continue

            self.show_clusters()    
        
if __name__ == "__main__":
    orchestrator = Orchestrator(sample_size=50000)
    orchestrator.run()