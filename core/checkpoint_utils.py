import os
import json
import time
import numpy as np # Required by get_processed_files_from_database if it were to handle numpy in payload
# from qdrant_client import QdrantClient # Not needed directly if client is passed as arg

def save_checkpoint(collection_name, processed_files, checkpoint_dir="./image_retrieval_project/checkpoints"):
    """Save processing checkpoint to disk"""
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, f"{collection_name}_checkpoint.json")

        checkpoint_data = {
            "collection_name": collection_name,
            "processed_files": list(processed_files), # Convert set to list for JSON
            "checkpoint_timestamp": time.time(),
            "checkpoint_version": "1.0" # Basic versioning
        }

        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        print(f"[CHECKPOINT] Saved checkpoint for {len(processed_files)} processed files to {checkpoint_file}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save checkpoint for {collection_name}: {e}")
        return False

def load_checkpoint(collection_name, checkpoint_dir="./image_retrieval_project/checkpoints"):
    """Load processing checkpoint from disk"""
    try:
        checkpoint_file = os.path.join(checkpoint_dir, f"{collection_name}_checkpoint.json")

        if not os.path.exists(checkpoint_file):
            print(f"[CHECKPOINT] No checkpoint found for collection {collection_name} at {checkpoint_file}")
            return set() # Return empty set if no checkpoint

        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)

        # Basic validation
        if checkpoint_data.get("collection_name") != collection_name:
            print(f"[WARNING] Checkpoint collection name mismatch: expected {collection_name}, found {checkpoint_data.get('collection_name')}")
            # Depending on strictness, might want to return empty set or raise error

        processed_files = set(checkpoint_data.get("processed_files", []))
        checkpoint_timestamp = checkpoint_data.get("checkpoint_timestamp", 0)

        print(f"[CHECKPOINT] Loaded checkpoint for {collection_name} with {len(processed_files)} files.")
        if checkpoint_timestamp:
            print(f"[CHECKPOINT] Checkpoint created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(checkpoint_timestamp))}")

        return processed_files
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint for {collection_name}: {e}")
        return set() # Return empty set on error

def get_processed_files_from_database(client, collection_name):
    """Query the database to get a list of already processed files based on 'source_filename' or 'filename' in payload."""
    if client is None:
        print("[ERROR] Qdrant client is None. Cannot query processed files.")
        return set()

    processed_files = set()
    try:
        print(f"[DB CHECK] Querying database for processed files in collection: {collection_name}...")

        # It's generally safer to scroll with a smaller limit and repeat if necessary,
        # but for filenames, a larger limit might be acceptable if payload is small.
        # Using a limit of 1000 and fetching in batches.
        offset = None
        while True:
            points, next_offset = client.scroll(
                collection_name=collection_name,
                limit=1000,
                offset=offset,
                with_payload=["source_filename", "filename", "full_path"] # Request specific fields
            )
            if not points:
                break

            for point in points:
                payload = point.payload
                if payload:
                    # Prioritize more specific fields if available
                    if "source_filename" in payload and payload["source_filename"]:
                        processed_files.add(os.path.basename(str(payload["source_filename"])))
                    elif "filename" in payload and payload["filename"]:
                        processed_files.add(os.path.basename(str(payload["filename"])))
                    elif "full_path" in payload and payload["full_path"]: # Fallback for older data
                        processed_files.add(os.path.basename(str(payload["full_path"])))

            if next_offset is None: # No more points to scroll
                break
            offset = next_offset
            print(f"[DB CHECK] Scrolled {len(processed_files)} processed file entries so far...")

        print(f"[DB CHECK] Found {len(processed_files)} unique processed filenames in database collection '{collection_name}'.")
    except Exception as e:
        # Specifically handle cases where the collection might not exist or is empty
        if "not found" in str(e).lower() or "doesn't exist" in str(e).lower():
            print(f"[DB CHECK] Collection '{collection_name}' not found or is empty. Returning 0 processed files.")
        else:
            print(f"[ERROR] Failed to query database for processed files in '{collection_name}': {e}")
            import traceback
            traceback.print_exc()
    return processed_files


def should_skip_file(filename, processed_files_checkpoint, processed_files_database):
    """Determine if a file should be skipped based on checkpoint and database state."""
    base_filename = os.path.basename(filename) # Ensure we are checking against basename

    in_checkpoint = base_filename in processed_files_checkpoint
    in_database = base_filename in processed_files_database

    should_skip = in_checkpoint or in_database

    if should_skip:
        source = "checkpoint" if in_checkpoint else "database"
        if in_checkpoint and in_database: source = "checkpoint and database"
        print(f"[SKIP] Skipping '{base_filename}' (already processed - found in {source})")

    return should_skip
