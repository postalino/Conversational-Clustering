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
)


# ---------------------------------------------------------------------------
# Action registry
# ---------------------------------------------------------------------------

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
        reason=args["reason"],
    ),
    "split_cluster": lambda self, args: split_cluster_state(
        data=self.data,
        cluster_metadata=self.cluster_metadata,
        cluster_id=args["cluster_id"],
        llm_service=self.llm_service,
        reason=args["reason"],
    ),
    "needs_clarification": lambda self, args: {
        "status": "needs_clarification",
        "question": args["question"],
        "reason": args["reason"],
    },
    "no_action": lambda self, args: {
        "status": "no_action",
        "reason": args["reason"],
    },
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

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
        self.turn_id = 0
        self.history = []

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def load_dataset(self) -> None:
        console.log("Loading dataset...")
        self.dataset.download_data()
        self.data = self.dataset.load_data(
            sample_size=self.sample_size,
            random_state=self.random_state,
        )
        console.log(f"Dataset loaded — {len(self.data):,} rows")

    def generate_embeddings(self) -> None:
        console.log("Generating embeddings...")
        cache_path = ROOT_DIR / "data" / "embeddings.npz"
        self.data = self.embedding_service.attach_embeddings_to_dataframe(
            df=self.data,
            cache_path=cache_path,
        )
        console.log("Embeddings ready")

    def cluster_data(self, method: str = "hdbscan", k: int | None = None) -> None:
        console.log(f"Clustering with {method}{f' (k={k})' if k else ''}...")
        self.embeddings = normalize(np.vstack(self.data["embedding"].to_numpy()))

        if method == "hdbscan":
            self.labels = cluster_with_hdbscan(self.embeddings)
        elif method == "kmeans":
            if k is None:
                raise ValueError("k must be provided for kmeans clustering")
            self.labels = cluster_with_kmeans(self.embeddings, k=k)
        else:
            raise ValueError(f"Invalid clustering method '{method}'. Use 'hdbscan' or 'kmeans'.")

        self.data["cluster_id"] = self.labels
        n_clusters = len(set(self.labels) - {-1})
        console.log(f"Found {n_clusters} clusters")

    def label_clusters(self) -> None:
        unique_ids = self.data["cluster_id"].unique()
        console.log(f"Labelling {len(unique_ids)} clusters...")

        for cluster_id in unique_ids:
            if cluster_id == -1:
                self.cluster_metadata[cluster_id] = {
                    "name": "Noise / Outliers",
                    "description": "Items not assigned to a stable cluster.",
                    "evidence": [],
                }
                continue

            console.log(f"   → Cluster {cluster_id}...")
            examples = get_representative_examples(data=self.data, cluster_id=cluster_id)
            self.cluster_metadata[cluster_id] = name_cluster(
                llm_service=self.llm_service, examples=examples
            )

        console.log("Clusters labelled")

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def build_displayed_state(self) -> dict:
        """Return a serialisable snapshot of the current cluster state."""
        davies_bouldin = calculate_all_davies_bouldin(self.data)

        clusters = []
        for cluster_id, meta in sorted(self.cluster_metadata.items()):
            count = int((self.data["cluster_id"] == cluster_id).sum())
            examples = get_representative_examples(
                data=self.data,
                cluster_id=cluster_id,
                samples_per_tercile=2,
                seed=42,
            )
            clusters.append({
                "cluster_id": int(cluster_id),
                "name": meta["name"],
                "description": meta["description"],
                "count": count,
                "davies_bouldin": davies_bouldin.get(cluster_id),
                "examples_shown": examples,
            })

        return {"clusters": clusters}

    def show_clusters(self) -> None:
        """Pretty-print the current cluster state to the console."""
        console.rule("[bold blue]Current clusters")
        state = self.build_displayed_state()

        for cluster in state["clusters"]:
            cluster_id = cluster["cluster_id"]
            meta = self.cluster_metadata[cluster_id]

            header = Text()
            header.append(f"Cluster {cluster_id}", style="bold cyan")
            header.append(" · ")
            header.append(cluster["name"], style="bold white")

            info_table = Table.grid(padding=(0, 2))
            info_table.add_column(style="magenta", justify="right")
            info_table.add_column(style="white")
            info_table.add_row("Description", cluster["description"])
            info_table.add_row("Items", str(cluster["count"]))

            db = cluster["davies_bouldin"]
            info_table.add_row("Davies-Bouldin", f"{db:.3f}" if db is not None else "N/A")

            examples_table = Table(
                title="Representative examples",
                show_header=True,
                header_style="bold green",
            )
            examples_table.add_column("Id", style="yellow", width=8)
            examples_table.add_column("Text", style="white")
            for ex in cluster["examples_shown"]:
                examples_table.add_row(str(ex["Id"]), ex["Text"].replace("\n", " ").strip())

            console.print(Panel.fit(header, border_style="blue"))
            console.print(info_table)
            console.print(examples_table)
            console.print()

    def save_turn_history(
        self,
        displayed_state: dict,
        user_input: str,
        parsed_action: dict,
        result: dict,
    ) -> None:
        self.turn_id += 1
        self.history.append({
            "turn_id": self.turn_id,
            "displayed_state": displayed_state,
            "user_input": user_input,
            "parsed_action": parsed_action,
            "action_executed": parsed_action.get("tool_name"),
            "system_response": result,
            "status": result.get("status", "applied"),
        })

    # ------------------------------------------------------------------
    # Feedback loop
    # ------------------------------------------------------------------

    def apply_feedback(self, user_feedback: str) -> dict:
        displayed_state = self.build_displayed_state()

        parsed_action = parse_feedback(
            llm_service=self.llm_service,
            user_feedback=user_feedback,
            cluster_metadata=self.cluster_metadata,
        )

        action_name = parsed_action.get("tool_name")
        arguments = parsed_action.get("arguments", {})

        if action_name not in ACTION_REGISTRY:
            result = {"status": "error", "message": f"Unknown action: {action_name}"}
            self.save_turn_history(displayed_state, user_feedback, parsed_action, result)
            raise ValueError(f"Unknown action: {action_name}")

        result = ACTION_REGISTRY[action_name](self, arguments)
        self.save_turn_history(displayed_state, user_feedback, parsed_action, result)
        return result

    def run(self) -> None:
        self.load_dataset()
        self.generate_embeddings()
        self.cluster_data(method="kmeans", k=5)
        self.label_clusters()
        self.show_clusters()

        while True:
            user_input = input(
                "\nProvide feedback, or type 'quit' / 'exit' / 'q' to end: "
            ).strip()

            if user_input.lower() in {"quit", "exit", "stop", "q"}:
                print("Exiting interactive loop.")
                break

            if not user_input:
                print("Empty input. Please write feedback or type 'quit'.")
                continue

            try:
                result = self.apply_feedback(user_input)
            except Exception as e:
                print(f"\nCould not apply feedback: {e}")
                continue

            status = result.get("status")

            if status == "needs_clarification":
                print(f"\nClarification needed: {result['question']}")
                print(f"Reason: {result['reason']}")
                continue

            if status == "no_action":
                print(f"\nNo action taken — {result['reason']}")
                continue

            print("\nAction applied:")
            print(result)
            self.show_clusters()


if __name__ == "__main__":
    orchestrator = Orchestrator(sample_size=50000)
    orchestrator.run()