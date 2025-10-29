import os
import httpx
import yaml


async def download_dataset(config_path: str, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    dataset_name = tuple(config['datasets'].keys())[0]
    csv_url = config['datasets'][dataset_name]
    save_path = os.path.join(save_dir, f"{dataset_name}.csv")
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            r = await client.get(csv_url)
            r.raise_for_status()
            with open(save_path, "wb") as f:
                f.write(r.content)
        except httpx.HTTPError as e:
            print(
                f"❌ Failed to download CSV file '{dataset_name}' from {csv_url}."
                f"The file was not saved to {save_path}. Reason: {e}"
            )