from dataset import DatasetDownloader
import tiktoken


EMBEDDING_PRICES = {
    "text-embedding-3-small": 0.02,  # dollars per 1M tokens
    "text-embedding-3-large": 0.13,  # dollars per 1M tokens
}


def count_tokens(text: str, encoding) -> int:
    return len(encoding.encode(str(text)))


if __name__ == "__main__":
    dataset = DatasetDownloader()
    df = dataset.load_data(sample_size=50000, random_state=42)

    encoding = tiktoken.get_encoding("cl100k_base")

    total_tokens = df["Text"].apply(
        lambda text: count_tokens(text, encoding)
    ).sum()

    print(f"Numero recensioni: {len(df)}")
    print(f"Token totali stimati: {total_tokens:,}")

    for model_name, price_per_million in EMBEDDING_PRICES.items():
        total_cost = (total_tokens / 1_000_000) * price_per_million
        print(f"{model_name}: ${total_cost:.4f}")