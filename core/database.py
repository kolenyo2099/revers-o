import os
import time
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
import torch
import sys
import shutil
import traceback

def setup_qdrant(collection_name, vector_size, max_retries=5, retry_delay=3.0):
    """Setup Qdrant collection for storing image region embeddings with retry logic"""
    persist_path = "./image_retrieval_project/qdrant_data"
    os.makedirs(persist_path, exist_ok=True)

    for attempt in range(max_retries):
        try:
            if attempt > 0: print(f"Retry attempt {attempt}/{max_retries} connecting to Qdrant...")
            client = QdrantClient(path=persist_path)
            print(f"Successfully connected to local storage at {persist_path}")
            try:
                collections = client.get_collections().collections
                collection_names = [collection.name for collection in collections]
                if collection_name not in collection_names:
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
                        optimizers_config=models.OptimizersConfigDiff(memmap_threshold=10000)
                    )
                    print(f"✅ Created new collection: {collection_name}")
                else:
                    print(f"✅ Using existing collection: {collection_name}")
                    try:
                        collection_info = client.get_collection(collection_name=collection_name)
                        existing_vector_size = collection_info.config.params.vectors.size
                        if existing_vector_size != vector_size:
                            print(f"⚠️ Warning: Collection {collection_name} has vector size {existing_vector_size}, but requested {vector_size}")
                    except Exception as inner_e: print(f"⚠️ Warning: Could not verify vector size: {inner_e}")
                return client
            except Exception as inner_e:
                print(f"Error working with collection: {inner_e}"); client.close()
                if attempt < max_retries - 1: time.sleep(retry_delay); continue
        except RuntimeError as e:
            if "already accessed by another instance" in str(e): # Specific check for lock
                print(f"⚠️ Database is locked. Waiting {retry_delay}s before retry {attempt+1}/{max_retries}...")
                if attempt < max_retries - 1: time.sleep(retry_delay)
                else: print(f"❌ Failed to access locked database after {max_retries} attempts."); return None
            else: print(f"❌ Error connecting to Qdrant: {e}"); traceback.print_exc(); return None
        except Exception as e: print(f"❌ Unexpected error in setup_qdrant: {e}"); traceback.print_exc(); return None
    return None

def store_embeddings_in_qdrant(client, collection_name, embeddings, metadata_list):
    if client is None: print(f"❌ DB client not connected"); return []
    if not embeddings or not metadata_list: print(f"❌ Nothing to store"); return []
    try:
        embedding_vectors = []
        for emb in embeddings:
            emb_numpy = emb.detach().cpu().numpy() if isinstance(emb, torch.Tensor) else np.array(emb)
            if len(emb_numpy.shape) == 1: emb_vector = emb_numpy
            elif len(emb_numpy.shape) == 2 and emb_numpy.shape[0] == 1: emb_vector = emb_numpy[0]
            else: emb_vector = emb_numpy.flatten()
            embedding_vectors.append(emb_vector)

        point_ids = [metadata["region_id"] for metadata in metadata_list]
        sanitized_metadata = []
        for metadata in metadata_list:
            sanitized = {}
            for key, value in metadata.items():
                if isinstance(value, np.integer): sanitized[key] = int(value)
                elif isinstance(value, np.floating): sanitized[key] = float(value)
                elif isinstance(value, np.ndarray): sanitized[key] = value.tolist()
                elif isinstance(value, list):
                    sanitized[key] = [int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else x for x in value]
                else: sanitized[key] = value
            sanitized_metadata.append(sanitized)

        points = [models.PointStruct(id=pid, vector=vec.tolist(), payload=meta) for pid, vec, meta in zip(point_ids, embedding_vectors, sanitized_metadata)]
        if not points: print(f"⚠️ No valid points to store"); return []

        stored_count = 0
        for i in range(0, len(points), 100): # Batch size 100
            batch = points[i:i+100]
            try:
                client.upsert(collection_name=collection_name, points=batch, wait=True) # Added wait=True
                stored_count += len(batch)
            except Exception as batch_error: print(f"❌ Error storing batch: {batch_error}")
        print(f"✅ Stored {stored_count}/{len(points)} embeddings in {collection_name}")
        return point_ids
    except Exception as e: print(f"❌ Error in store_embeddings_in_qdrant: {e}"); traceback.print_exc(); return []

def list_available_databases():
    collections = []
    qdrant_data_dir = os.path.join("image_retrieval_project", "qdrant_data")
    if os.path.exists(qdrant_data_dir):
        try:
            client = QdrantClient(path=qdrant_data_dir)
            collections = [c.name for c in client.get_collections().collections]
            client.close()
        except Exception as e:
            print(f"[DEBUG] Error querying Qdrant API for collections: {e}")
            # Fallback to filesystem checks
            for d_name in ["collection", "collections"]: # Check both common names
                check_dir = os.path.join(qdrant_data_dir, d_name)
                if os.path.exists(check_dir) and os.path.isdir(check_dir): # Ensure it's a directory
                    collections.extend([d for d in os.listdir(check_dir) if os.path.isdir(os.path.join(check_dir, d))])

    unique_collections = sorted(list(set(c for c in collections if not c.startswith(".")))) # Filter hidden and ensure unique
    print(f"[STATUS] Available collections: {unique_collections}" if unique_collections else "[STATUS] No database collections found")
    return unique_collections

def delete_database_collection(collection_name):
    if not collection_name or collection_name == "No databases found": return False, "No valid collection specified"
    try:
        qdrant_data_dir = os.path.join("image_retrieval_project", "qdrant_data")
        client = QdrantClient(path=qdrant_data_dir)
        # Check if collection exists before attempting deletion
        collection_names = [c.name for c in client.get_collections().collections]
        if collection_name not in collection_names:
            client.close()
            return False, f"Collection '{collection_name}' does not exist"

        client.delete_collection(collection_name=collection_name)
        client.close()
        print(f"[STATUS] Successfully deleted collection: {collection_name}")
        return True, f"Successfully deleted database: {collection_name}"
    except Exception as e:
        print(f"[ERROR] Failed to delete collection {collection_name}: {e}"); traceback.print_exc()
        return False, f"Error deleting database: {str(e)}"

def cleanup_qdrant_connections(force=False):
    """Attempt to forcefully clean up any existing Qdrant connections"""
    qdrant_data_dir = os.path.join("image_retrieval_project", "qdrant_data")
    lock_file = os.path.join(qdrant_data_dir, ".lock")
    os.makedirs(qdrant_data_dir, exist_ok=True)

    if force and sys.platform in ["darwin", "linux"]:
        try:
            import subprocess
            cmd = f"lsof -t +D \"{qdrant_data_dir}\" 2>/dev/null || echo ''"
            result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
            pids_str = result.stdout.strip()
            if pids_str:
                pids = [pid for pid in pids_str.split("\n") if pid and pid.strip() and int(pid) != os.getpid()]
                if pids:
                    print(f"[CLEANUP] Found {len(pids)} external process(es) using Qdrant data: {', '.join(pids)}. Terminating...")
                    subprocess.run(f"kill -9 {' '.join(pids)}", shell=True, check=False)
                    time.sleep(0.5)
            else: print(f"[CLEANUP] No external processes found using Qdrant data directory.")
        except Exception as e: print(f"[WARNING] Error during process cleanup: {e}")

    lock_removed = False
    if os.path.exists(lock_file):
        print(f"[CLEANUP] Found Qdrant lock file: {lock_file}. Attempting removal...")
        try:
            os.remove(lock_file); lock_removed = True
            print(f"[CLEANUP] Successfully removed lock file.")
            time.sleep(0.5)
        except Exception as e:
            print(f"[WARNING] Could not remove lock file {lock_file}: {e}")
            if force:
                print(f"[CLEANUP] Attempting forced removal of {lock_file}...")
                try:
                    cmd = f"rm -f \"{lock_file}\"" if sys.platform in ["darwin", "linux"] else f"del /F \"{lock_file}\""
                    subprocess.run(cmd, shell=True, check=True)
                    if not os.path.exists(lock_file): lock_removed = True; print(f"[CLEANUP] Successfully force-removed lock file."); time.sleep(0.5)
                except Exception as force_e: print(f"[WARNING] Force removal of lock file failed: {force_e}")
    else: print(f"[CLEANUP] No Qdrant lock file found at {lock_file}."); lock_removed = True

    if force and not lock_removed:
         print(f"[WARNING] Lock file {lock_file} may still exist despite forced removal attempts.")
    return lock_removed

def verify_repair_database(collection_name, force_rebuild=False):
    """Verifies and optionally repairs a database collection."""
    print(f"[DB MAINT] Verifying collection: {collection_name}, Force rebuild: {force_rebuild}")
    qdrant_data_dir = os.path.join("image_retrieval_project", "qdrant_data")
    client = None
    try:
        client = QdrantClient(path=qdrant_data_dir)
        collections_response = client.get_collections()
        collection_names = [c.name for c in collections_response.collections]

        if collection_name not in collection_names:
            return False, f"Collection '{collection_name}' not found."

        collection_info = client.get_collection(collection_name=collection_name)
        vector_size = collection_info.config.params.vectors.size
        point_count_response = client.count(collection_name=collection_name, exact=True)
        point_count = point_count_response.count
        print(f"[DB MAINT] Collection '{collection_name}': {point_count} points, vector size {vector_size}.")

        if point_count == 0 and not force_rebuild:
            return True, f"Collection '{collection_name}' is empty but valid."

        if force_rebuild:
            print(f"[DB MAINT] Force rebuilding collection '{collection_name}'...")
            all_points = []
            offset = None
            print(f"[DB MAINT] Reading all points from '{collection_name}' for backup...")
            while True:
                scroll_res, next_offset = client.scroll(
                    collection_name=collection_name, limit=200, offset=offset,
                    with_payload=True, with_vectors=True)
                if not scroll_res: break
                all_points.extend(scroll_res)
                if not next_offset: break
                offset = next_offset
            print(f"[DB MAINT] Read {len(all_points)} points for backup.")

            print(f"[DB MAINT] Deleting original collection '{collection_name}' for rebuild...")
            client.delete_collection(collection_name=collection_name)
            time.sleep(0.5)

            print(f"[DB MAINT] Recreating collection '{collection_name}'...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
                optimizers_config=models.OptimizersConfigDiff(memmap_threshold=10000)
            )

            if all_points:
                points_to_upsert = [models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in all_points]
                # Upsert in batches
                batch_size_rebuild = 100
                for i in range(0, len(points_to_upsert), batch_size_rebuild):
                    batch = points_to_upsert[i:i+batch_size_rebuild]
                    client.upsert(collection_name=collection_name, points=batch, wait=True)
                print(f"[DB MAINT] Restored {len(points_to_upsert)} points to '{collection_name}'.")
            else: print(f"[DB MAINT] No points were in the original collection to restore.")

            new_point_count_response = client.count(collection_name=collection_name, exact=True)
            new_point_count = new_point_count_response.count
            if new_point_count == len(all_points):
                return True, f"Collection '{collection_name}' successfully rebuilt. {new_point_count} points restored."
            else:
                return False, f"Collection '{collection_name}' rebuild issue. Expected {len(all_points)}, found {new_point_count}."

        if point_count > 0: # Basic validation if not rebuilding
            print(f"[DB MAINT] Performing sample point check...")
            try:
                sample_points, _ = client.scroll(collection_name=collection_name, limit=min(5, point_count), with_payload=True)
                if not sample_points and point_count > 0 : # Check if scroll returned empty despite count
                     return False, f"Inconsistency: {point_count} points counted, but no points retrieved."
                print(f"[DB MAINT] Sample check OK: Retrieved {len(sample_points)} sample points.")
            except Exception as e_scroll:
                return False, f"Failed to sample points: {e_scroll}"

        return True, f"Collection '{collection_name}' verified. {point_count} points found."
    except Exception as e:
        print(f"[ERROR] Database verification/repair for '{collection_name}' failed: {e}"); traceback.print_exc()
        return False, f"Verification/repair failed: {str(e)}"
    finally:
        if client: client.close()
        print(f"[DB MAINT] Closed Qdrant client for {collection_name}.")
