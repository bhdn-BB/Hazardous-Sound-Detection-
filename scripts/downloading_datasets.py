import os
import httpx


async def download_dataset(
        save_dir: str,
        url_dataset: str,
        dataset_name: str
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{dataset_name}.csv")
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            r = await client.get(url_dataset)
            r.raise_for_status()
            with open(save_path, "wb") as f:
                f.write(r.content)
        except httpx.HTTPError as e:
            print(
                f"❌ Failed to download CSV file '{dataset_name}' from {url_dataset}. "
                f"The file was not saved to {save_path}. Reason: {e}"
            )