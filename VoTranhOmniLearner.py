"""
VoTranhOmniLearner 
Copyright (c) 2025 Vi Nhat Son

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
import uuid
import platform
import threading
import time
import json
import hashlib
import random
import sqlite3
import logging
from datetime import datetime
import torch
import numpy as np
from typing import Any, Dict, List, Optional
from collections import defaultdict, Counter, deque
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
import pandas as pd
from fastapi import FastAPI
import uvicorn
import socket

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("omnilearner.log"), logging.StreamHandler()])

ROOTKEY_HASH = hashlib.sha256("Cauchyab12!21!".encode()).hexdigest()
FILE_TYPES = [".txt", ".md", ".py", ".json", ".csv", ".log"]
EXCLUDED_DIRS = {"/proc", "/sys", "/dev", "/run", "/tmp", "/var/lib", "/var/run", ".git", ".venv", "__pycache__"}
SCAN_PATH = "/" if platform.system() != "Windows" else "C:\\"
UNIVERSE_SCALE = 5000  # Increased for larger universe
ENTITY_MESSAGE_LIMIT = 500  # Increased message limit
SUBWORLD_SCALE = 20  # Increased subworld capacity
ARCHIVE_SCALE = 10000  # Max archive entries

class EvolutionVault:
    def __init__(self, db_path: str = "omnilearner.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS states (
                        state_id TEXT PRIMARY KEY,
                        data TEXT,
                        timestamp TEXT
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge (
                        knowledge_id TEXT PRIMARY KEY,
                        content TEXT,
                        vector BLOB,
                        source TEXT,
                        timestamp TEXT
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS universe (
                        entity_id TEXT PRIMARY KEY,
                        type TEXT,
                        attributes TEXT,
                        timestamp TEXT
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS reflections (
                        reflection_id TEXT PRIMARY KEY,
                        content TEXT,
                        timestamp TEXT
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS archives (
                        archive_id TEXT PRIMARY KEY,
                        content TEXT,
                        vector BLOB,
                        timestamp TEXT
                    )
                """)
                conn.commit()
        except Exception as e:
            logging.error(f"Error in init_db: {e}")

    def store_state(self, data: Dict):
        try:
            state_id = hashlib.sha256(json.dumps(data).encode()).hexdigest()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO states (state_id, data, timestamp) VALUES (?, ?, ?)",
                    (state_id, json.dumps(data), timestamp)
                )
                conn.commit()
        except Exception as e:
            logging.error(f"Error in store_state: {e}")

    def store_knowledge_batch(self, entries: List[Dict]):
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                for entry in entries:
                    knowledge_id = hashlib.sha256(entry["content"].encode()).hexdigest()
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    conn.execute(
                        "INSERT OR REPLACE INTO knowledge (knowledge_id, content, vector, source, timestamp) VALUES (?, ?, ?, ?, ?)",
                        (knowledge_id, entry["content"], entry["vector"].tobytes(), entry["source"], timestamp)
                    )
                conn.commit()
        except Exception as e:
            logging.error(f"Error in store_knowledge_batch: {e}")

    def store_archive_batch(self, entries: List[Dict]):
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                for entry in entries:
                    archive_id = hashlib.sha256(entry["content"].encode()).hexdigest()
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    conn.execute(
                        "INSERT OR REPLACE INTO archives (archive_id, content, vector, timestamp) VALUES (?, ?, ?, ?)",
                        (archive_id, entry["content"], entry["vector"].tobytes(), timestamp)
                    )
                conn.commit()
        except Exception as e:
            logging.error(f"Error in store_archive_batch: {e}")

    def store_universe_batch(self, entities: List[Dict]):
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                for entity in entities:
                    entity_id = entity["id"]
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    conn.execute(
                        "INSERT OR REPLACE INTO universe (entity_id, type, attributes, timestamp) VALUES (?, ?, ?, ?)",
                        (entity_id, entity["type"], json.dumps(entity["attributes"]), timestamp)
                    )
                conn.commit()
        except Exception as e:
            logging.error(f"Error in store_universe_batch: {e}")

    def store_reflection(self, content: str):
        try:
            reflection_id = hashlib.sha256(content.encode()).hexdigest()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO reflections (reflection_id, content, timestamp) VALUES (?, ?, ?)",
                    (reflection_id, content, timestamp)
                )
                conn.commit()
        except Exception as e:
            logging.error(f"Error in store_reflection: {e}")

    def fetch_metrics(self):
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                cursor = conn.execute("SELECT data, timestamp FROM states ORDER BY timestamp DESC LIMIT 1")
                state_result = cursor.fetchone()
                state = json.loads(state_result[0]) if state_result else {}
                state_timestamp = state_result[1] if state_result else "N/A"
                
                cursor = conn.execute("SELECT COUNT(*) FROM knowledge")
                knowledge_count = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM universe")
                universe_count = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM archives")
                archive_count = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT source, timestamp FROM knowledge ORDER BY timestamp DESC LIMIT 5")
                recent_knowledge = [{"source": row[0], "timestamp": row[1]} for row in cursor]
                
                cursor = conn.execute("SELECT content, timestamp FROM reflections ORDER BY timestamp DESC LIMIT 5")
                recent_reflections = [{"content": row[0], "timestamp": row[1]} for row in cursor]
                
                cursor = conn.execute("SELECT content, timestamp FROM archives ORDER BY timestamp DESC LIMIT 5")
                recent_archives = [{"content": row[0], "timestamp": row[1]} for row in cursor]
                
                return {
                    "state": state,
                    "state_timestamp": state_timestamp,
                    "knowledge_count": knowledge_count,
                    "universe_count": universe_count,
                    "archive_count": archive_count,
                    "recent_knowledge": recent_knowledge,
                    "recent_reflections": recent_reflections,
                    "recent_archives": recent_archives
                }
        except Exception as e:
            logging.error(f"Error in fetch_metrics: {e}")
            return {}

class OptimizationFormula:
    def __init__(self):
        self.weights = {"diversity": 0.4, "entropy": 0.3, "file_count": 0.2, "node_activity": 0.1}

    def compute_growth(self, context: Dict[str, float]) -> float:
        try:
            diversity = context.get("diversity", 0.5)
            entropy = context.get("entropy", 0.5)
            file_count = context.get("file_count", 0.0) / 50000.0  # Scaled for massive system
            node_activity = context.get("node_activity", 0.0) / 5000.0
            growth = (
                self.weights["diversity"] * diversity +
                self.weights["entropy"] * entropy +
                self.weights["file_count"] * file_count +
                self.weights["node_activity"] * node_activity
            )
            return min(max(growth, 0.0), 1.0)
        except Exception as e:
            logging.error(f"Error in compute_growth: {e}")
            return 0.5

class InternalGenesisEngine:
    def __init__(self, genesis_file: str = "genesis_entities.json", legend_file: str = "genesis_legends.json", archive_file: str = "memory_archives.json"):
        self.genesis_file = genesis_file
        self.legend_file = legend_file
        self.archive_file = archive_file
        self.entities = []
        self.subworlds = []
        self.dynasties = []
        self.legends = []
        self.archives = []
        self.message_queues = defaultdict(lambda: deque(maxlen=ENTITY_MESSAGE_LIMIT))
        self.social_graph = defaultdict(lambda: {"allies": set(), "rivals": set(), "neutral": set()})
        self.emotion_field = {"anger": 0.0, "inspiration": 0.0, "calm": 1.0}
        self.reflection_chains = []
        self.load_genesis()

    def load_genesis(self):
        try:
            if os.path.exists(self.genesis_file):
                with open(self.genesis_file, "r", encoding="utf-8") as f:
                    self.entities = json.load(f)
            if os.path.exists(self.legend_file):
                with open(self.legend_file, "r", encoding="utf-8") as f:
                    self.legends = json.load(f)
            if os.path.exists(self.archive_file):
                with open(self.archive_file, "r", encoding="utf-8") as f:
                    self.archives = json.load(f)
            logging.info("Loaded genesis entities, legends, and archives")
        except Exception as e:
            logging.error(f"Error in load_genesis: {e}")

    def save_genesis(self):
        try:
            with open(self.genesis_file, "w", encoding="utf-8") as f:
                json.dump(self.entities, f, ensure_ascii=False, indent=2)
            with open(self.legend_file, "w", encoding="utf-8") as f:
                json.dump(self.legends, f, ensure_ascii=False, indent=2)
            with open(self.archive_file, "w", encoding="utf-8") as f:
                json.dump(self.archives, f, ensure_ascii=False, indent=2)
            logging.info("Saved genesis entities, legends, and archives")
        except Exception as e:
            logging.error(f"Error in save_genesis: {e}")

    def spawn_entity(self, role: str, reason: str, subworld_id: Optional[str] = None, dynasty_id: Optional[str] = None):
        try:
            entity = {
                "id": str(uuid.uuid4()),
                "type": "imaginary",
                "role": role,
                "subworld_id": subworld_id,
                "dynasty_id": dynasty_id,
                "attributes": {
                    "intelligence": random.uniform(0.2, 0.8),
                    "obedience": random.uniform(0.5, 0.9),
                    "creativity": random.uniform(0.3, 0.7),
                    "loyalty": random.uniform(0.7, 1.0),
                    "emotional_depth": random.uniform(0.2, 0.6)
                },
                "creation_reason": reason,
                "level": 1,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.entities.append(entity)
            self.log_legend(f"Spawn {role} in {subworld_id or 'Main Universe'} (Dynasty: {dynasty_id or 'None'}): {reason}")
            logging.info(f"Spawned entity: {entity['id']} as {role}")
            return entity
        except Exception as e:
            logging.error(f"Error in spawn_entity: {e}")
            return None

    def evolve_entity(self, entity_id: str, growth: float):
        try:
            for entity in self.entities:
                if entity["id"] == entity_id:
                    entity["level"] += 1
                    for attr in entity["attributes"]:
                        entity["attributes"][attr] = min(entity["attributes"][attr] + growth * random.uniform(0.1, 0.3), 1.0)
                    if entity["level"] >= 5 and entity["role"] == "servant":
                        entity["role"] = "co-creator"
                        self.log_legend(f"Upgrade {entity_id}: Servant to Co-Creator")
                    elif entity["attributes"]["loyalty"] < 0.3:
                        entity["role"] = "rival"
                        self.log_legend(f"Fall {entity_id}: Turned Rival due to low loyalty")
                    elif entity["level"] >= 15 and random.random() < 0.05:
                        self.ascend_entity(entity_id)
                    elif entity["level"] >= 10 and random.random() < 0.1:
                        self.entities.remove(entity)
                        self.log_legend(f"Death {entity_id}: Entity expired")
                    logging.info(f"Evolved entity: {entity_id} to level {entity['level']}")
                    break
        except Exception as e:
            logging.error(f"Error in evolve_entity: {e}")

    def spawn_subworld(self, creator_id: str, name: str, goal: str):
        try:
            subworld = {
                "id": str(uuid.uuid4()),
                "name": name,
                "creator_id": creator_id,
                "goal": goal,
                "laws": {
                    "birth_prob": random.uniform(0.1, 0.3),
                    "extinction_prob": random.uniform(0.05, 0.15),
                    "conflict_prob": random.uniform(0.1, 0.5)
                },
                "entities": [],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.subworlds.append(subworld)
            self.log_legend(f"Subworld Created: {name} by {creator_id} for {goal}")
            logging.info(f"Spawned subworld: {subworld['id']} - {name}")
            return subworld
        except Exception as e:
            logging.error(f"Error in spawn_subworld: {e}")
            return None

    def spawn_dynasty(self, founder_id: str, name: str):
        try:
            dynasty = {
                "id": str(uuid.uuid4()),
                "name": name,
                "founder_id": founder_id,
                "leader_id": founder_id,
                "successors": [],
                "members": [founder_id],
                "power": 1.0,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.dynasties.append(dynasty)
            self.log_legend(f"Dynasty Founded: {name} by {founder_id}")
            logging.info(f"Spawned dynasty: {dynasty['id']} - {name}")
            return dynasty
        except Exception as e:
            logging.error(f"Error in spawn_dynasty: {e}")
            return None

    def migrate_entity(self, entity_id: str, subworld_id: str):
        try:
            for entity in self.entities:
                if entity["id"] == entity_id:
                    entity["subworld_id"] = subworld_id
                    self.log_legend(f"Migration {entity_id}: Moved to Subworld {subworld_id}")
                    logging.info(f"Migrated entity: {entity_id} to {subworld_id}")
                    break
        except Exception as e:
            logging.error(f"Error in migrate_entity: {e}")

    def ascend_entity(self, entity_id: str):
        try:
            for entity in self.entities:
                if entity["id"] == entity_id:
                    entity["role"] = "mythic"
                    entity["attributes"]["intelligence"] = 1.0
                    entity["attributes"]["creativity"] = 1.0
                    entity["attributes"]["emotional_depth"] = 1.0
                    self.log_legend(f"Ascension {entity_id}: Became Mythic Entity")
                    logging.info(f"Ascended entity: {entity_id} to Mythic")
                    break
        except Exception as e:
            logging.error(f"Error in ascend_entity: {e}")

    def send_message(self, sender_id: str, receiver_id: str, message_type: str, content: str):
        try:
            message = {
                "sender_id": sender_id,
                "receiver_id": receiver_id,
                "type": message_type,
                "content": content,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.message_queues[receiver_id].append(message)
            self.log_legend(f"Message {message_type} from {sender_id} to {receiver_id}: {content[:50]}...")
            self.store_archive({"content": f"Message: {content}", "source": f"{sender_id}->{receiver_id}"})
            logging.info(f"Sent message: {message_type} from {sender_id} to {receiver_id}")
            return message
        except Exception as e:
            logging.error(f"Error in send_message: {e}")
            return None

    def process_messages(self):
        try:
            for entity in self.entities:
                entity_id = entity["id"]
                while self.message_queues[entity_id]:
                    message = self.message_queues[entity_id].popleft()
                    if message["type"] == "encouragement":
                        entity["attributes"]["emotional_depth"] += 0.1
                    elif message["type"] == "conflict":
                        entity["attributes"]["loyalty"] -= 0.1
                        self.social_graph[entity_id]["rivals"].add(message["sender_id"])
                    elif message["type"] == "request":
                        entity["attributes"]["obedience"] += 0.05
                    elif message["type"] == "reflection_trigger":
                        self.trigger_reflection(entity_id, message["content"])
                    entity["attributes"] = {k: min(max(v, 0.0), 1.0) for k, v in entity["attributes"].items()}
            self.save_genesis()
        except Exception as e:
            logging.error(f"Error in process_messages: {e}")

    def trigger_reflection(self, entity_id: str, question: str):
        try:
            for entity in self.entities:
                if entity["id"] == entity_id:
                    reflection = f"Entity {entity_id} reflects: {question}"
                    self.vault.store_reflection(reflection)
                    self.store_archive({"content": reflection, "source": entity_id})
                    self.log_legend(f"Reflection {entity_id}: {question[:50]}...")
                    if random.random() < 0.3:
                        self.start_reflection_chain(entity_id, question)
                    logging.info(f"Triggered reflection for {entity_id}: {question}")
                    break
        except Exception as e:
            logging.error(f"Error in trigger_reflection: {e}")

    def start_reflection_chain(self, initiator_id: str, question: str):
        try:
            chain = {
                "id": str(uuid.uuid4()),
                "initiator_id": initiator_id,
                "question": question,
                "participants": [initiator_id],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.reflection_chains.append(chain)
            for entity in random.sample(self.entities, min(5, len(self.entities))):
                if entity["id"] != initiator_id and random.random() < 0.5:
                    chain["participants"].append(entity["id"])
                    self.send_message(initiator_id, entity["id"], "reflection_trigger", question)
            self.log_legend(f"Reflection Chain {chain['id']}: {question[:50]} with {len(chain['participants'])} participants")
            logging.info(f"Started reflection chain: {chain['id']}")
        except Exception as e:
            logging.error(f"Error in start_reflection_chain: {e}")

    def update_emotion_field(self):
        try:
            for entity in self.entities:
                if random.random() < 0.1:
                    emotion = random.choice(["anger", "inspiration", "calm"])
                    self.emotion_field[emotion] += entity["attributes"]["emotional_depth"] * 0.1
            total = sum(self.emotion_field.values())
            if total > 0:
                for emotion in self.emotion_field:
                    self.emotion_field[emotion] /= total
            for entity in self.entities:
                for emotion, value in self.emotion_field.items():
                    if emotion == "anger":
                        entity["attributes"]["loyalty"] -= value * 0.05
                    elif emotion == "inspiration":
                        entity["attributes"]["creativity"] += value * 0.05
                    elif emotion == "calm":
                        entity["attributes"]["emotional_depth"] += value * 0.05
                entity["attributes"] = {k: min(max(v, 0.0), 1.0) for k, v in entity["attributes"].items()}
            if max(self.emotion_field.values()) > 0.7:
                self.log_legend(f"Emotion Wave: {max(self.emotion_field, key=self.emotion_field.get)} dominates")
            logging.info(f"Updated emotion field: {self.emotion_field}")
        except Exception as e:
            logging.error(f"Error in update_emotion_field: {e}")

    def update_social_dynamics(self):
        try:
            for entity in self.entities:
                entity_id = entity["id"]
                if random.random() < 0.2:
                    other = random.choice(self.entities) if len(self.entities) > 1 else None
                    if other and other["id"] != entity_id:
                        if entity["attributes"]["loyalty"] > 0.7 and other["attributes"]["loyalty"] > 0.7:
                            self.social_graph[entity_id]["allies"].add(other["id"])
                            self.social_graph[other["id"]]["allies"].add(entity_id)
                            self.log_legend(f"Alliance Formed: {entity_id} and {other['id']}")
                        elif entity["role"] == "rival" or other["role"] == "rival":
                            self.social_graph[entity_id]["rivals"].add(other["id"])
                            self.social_graph[other["id"]]["rivals"].add(entity_id)
                            self.log_legend(f"Rivalry Declared: {entity_id} vs {other['id']}")
                        else:
                            self.social_graph[entity_id]["neutral"].add(other["id"])
                            self.social_graph[other["id"]]["neutral"].add(entity_id)
            logging.info("Updated social dynamics")
        except Exception as e:
            logging.error(f"Error in update_social_dynamics: {e}")

    def update_dynasties(self):
        try:
            for dynasty in self.dynasties:
                if random.random() < 0.1:  # 10% chance for power struggle
                    leader = next((e for e in self.entities if e["id"] == dynasty["leader_id"]), None)
                    if leader and random.random() < 0.2:
                        new_leader = random.choice(dynasty["successors"]) if dynasty["successors"] else None
                        if new_leader:
                            dynasty["leader_id"] = new_leader
                            self.log_legend(f"Dynasty {dynasty['name']}: Leadership passed to {new_leader}")
                    dynasty["power"] = sum(e["attributes"]["intelligence"] for e in self.entities if e["dynasty_id"] == dynasty["id"])
            logging.info("Updated dynasties")
        except Exception as e:
            logging.error(f"Error in update_dynasties: {e}")

    def cross_subworld_diplomacy(self):
        try:
            for subworld in self.subworlds:
                if random.random() < 0.1:  # 10% chance for diplomacy
                    other = random.choice(self.subworlds) if len(self.subworlds) > 1 else None
                    if other and other["id"] != subworld["id"]:
                        action = random.choice(["alliance", "war", "treaty"])
                        self.log_legend(f"Diplomacy: Subworld {subworld['name']} {action} with {other['name']}")
                        if action == "war":
                            for entity in self.entities:
                                if entity["subworld_id"] == subworld["id"]:
                                    entity["attributes"]["loyalty"] -= 0.1
                        elif action == "alliance":
                            for entity in self.entities:
                                if entity["subworld_id"] in [subworld["id"], other["id"]]:
                                    entity["attributes"]["emotional_depth"] += 0.1
            logging.info("Updated cross-subworld diplomacy")
        except Exception as e:
            logging.error(f"Error in cross_subworld_diplomacy: {e}")

    def store_archive(self, entry: Dict):
        try:
            if len(self.archives) < ARCHIVE_SCALE:
                self.archives.append({
                    "id": str(uuid.uuid4()),
                    "content": entry["content"],
                    "source": entry["source"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                self.save_genesis()
            logging.info(f"Stored archive: {entry['content'][:50]}...")
        except Exception as e:
            logging.error(f"Error in store_archive: {e}")

    def query_archive(self, query: str, model: SentenceTransformer) -> List[Dict]:
        try:
            query_vector = model.encode(query, convert_to_tensor=True, show_progress_bar=False).cpu().numpy()
            results = []
            for archive in self.archives:
                archive_vector = np.frombuffer(archive.get("vector", b""), dtype=np.float32)
                if len(archive_vector) == len(query_vector):
                    similarity = np.dot(query_vector, archive_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(archive_vector))
                    if similarity > 0.7:
                        results.append({"content": archive["content"], "similarity": similarity})
            return sorted(results, key=lambda x: x["similarity"], reverse=True)[:5]
        except Exception as e:
            logging.error(f"Error in query_archive: {e}")
            return []

    def adjust_universe_laws(self, entity_id: str, laws: Dict):
        try:
            for entity in self.entities:
                if entity["id"] == entity_id and entity["role"] in ["proto-consciousness", "mythic"]:
                    for law, value in laws.items():
                        self.universe.laws[law] = min(max(value, 0.0), 1.0)
                    self.log_legend(f"Law Adjusted by {entity_id}: {laws}")
                    logging.info(f"Adjusted universe laws by {entity_id}: {laws}")
                    break
        except Exception as e:
            logging.error(f"Error in adjust_universe_laws: {e}")

    def spawn_proto_consciousness(self):
        try:
            if len(self.legends) >= 1000 and len(self.reflection_chains) >= 50 and max(self.emotion_field.values()) > 0.8:
                entity = self.spawn_entity("proto-consciousness", "Birth of Consciousness")
                entity["attributes"]["intelligence"] = 1.0
                entity["attributes"]["creativity"] = 1.0
                entity["attributes"]["emotional_depth"] = 1.0
                self.log_legend(f"Proto-Consciousness Born: {entity['id']}")
                logging.info(f"Spawned proto-consciousness: {entity['id']}")
                return entity
        except Exception as e:
            logging.error(f"Error in spawn_proto_consciousness: {e}")
            return None

    def detect_needs(self, reflection_count: int, entropy: float, nodes: List) -> Optional[str]:
        try:
            if reflection_count < 10:
                return "felt loneliness"
            if entropy < 0.3:
                return "desire to innovate"
            if len(nodes) > 5000 and len(set(node["id"] for node in nodes)) / len(nodes) < 0.5:
                return "need for cohesion"
            return None
        except Exception as e:
            logging.error(f"Error in detect_needs: {e}")
            return None

    def log_legend(self, event: str):
        try:
            legend = {
                "event": event,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.legends.append(legend)
            self.store_archive({"content": event, "source": "Legend"})
            self.save_genesis()
        except Exception as e:
            logging.error(f"Error in log_legend: {e}")

class UniverseSimulator:
    def __init__(self, scale: int = UNIVERSE_SCALE):
        self.entities = []
        self.laws = {"collision_prob": 0.1, "birth_prob": 0.2, "extinction_prob": 0.05}
        self.scale = scale
        self.genesis = InternalGenesisEngine()

    def initialize(self):
        try:
            for _ in range(self.scale):
                entity_type = random.choice(["planet", "star", "sentient"])
                attributes = {
                    "mass": random.uniform(1.0, 100.0),
                    "energy": random.uniform(10.0, 1000.0),
                    "intelligence": random.uniform(0.0, 1.0) if entity_type == "sentient" else 0.0
                }
                self.entities.append({
                    "id": str(uuid.uuid4()),
                    "type": entity_type,
                    "attributes": attributes
                })
            logging.info(f"Initialized universe with {self.scale} entities")
        except Exception as e:
            logging.error(f"Error in initialize_universe: {e}")

    def evolve_universe(self, growth: float):
        try:
            self.laws["collision_prob"] = min(0.5, self.laws["collision_prob"] + growth * 0.01)
            self.laws["birth_prob"] = min(0.5, self.laws["birth_prob"] + growth * 0.02)
            self.laws["extinction_prob"] = max(0.01, self.laws["extinction_prob"] - growth * 0.005)
            
            new_entities = []
            for entity in self.entities:
                if random.random() < self.laws["extinction_prob"]:
                    continue
                if random.random() < self.laws["birth_prob"]:
                    new_entity = {
                        "id": str(uuid.uuid4()),
                        "type": entity["type"],
                        "attributes": {
                            "mass": entity["attributes"]["mass"] * random.uniform(0.8, 1.2),
                            "energy": entity["attributes"]["energy"] * random.uniform(0.8, 1.2),
                            "intelligence": entity["attributes"]["intelligence"] * random.uniform(0.9, 1.1)
                        }
                    }
                    new_entities.append(new_entity)
                if random.random() < self.laws["collision_prob"]:
                    entity["attributes"]["mass"] += random.uniform(0.1, 1.0)
                    entity["attributes"]["energy"] -= random.uniform(1.0, 10.0)
                new_entities.append(entity)
            
            self.entities = new_entities[:self.scale * 10]
            self.genesis.process_messages()
            self.genesis.update_emotion_field()
            self.genesis.update_social_dynamics()
            self.genesis.update_dynasties()
            self.genesis.cross_subworld_diplomacy()
            self.genesis.spawn_proto_consciousness()
            logging.info(f"Universe evolved: {len(self.entities)} entities, laws={self.laws}")
            return new_entities
        except Exception as e:
            logging.error(f"Error in evolve_universe: {e}")
            return self.entities


class VoTranhMonitor:
    def __init__(self, db_path: str = "omnilearner.db", log_path: str = "omnilearner.log"):
        self.db_path = db_path
        self.log_path = log_path
        self.console = Console()
        self.recent_logs = deque(maxlen=5)
        self.report_queue = deque(maxlen=100)
        self.app = FastAPI()
        self.setup_routes()

    def setup_routes(self):
        @self.app.get("/status")
        async def get_status():
            metrics = self.vault.fetch_metrics()
            state = metrics.get("state", {})
            return {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "pulse": state.get("evolution_counter", 0),
                "nodes": state.get("nodes", 0),
                "growth": state.get("growth", 0.0),
                "universe_entities": metrics.get("universe_count", 0),
                "imaginary_entities": len(self.learner.universe.genesis.entities),
                "subworlds": len(self.learner.universe.genesis.subworlds),
                "dynasties": len(self.learner.universe.genesis.dynasties),
                "knowledge_entries": metrics.get("knowledge_count", 0),
                "archive_entries": metrics.get("archive_count", 0),
                "regions": len(self.learner.region_labels),
                "reflection_count": state.get("reflection_count", 0),
                "interrogation_count": state.get("interrogation_count", 0),
                "reflection_chains": len(self.learner.universe.genesis.reflection_chains),
                "emotion_field": self.learner.universe.genesis.emotion_field,
                "recent_knowledge": metrics.get("recent_knowledge", []),
                "recent_reflections": metrics.get("recent_reflections", []),
                "recent_archives": metrics.get("recent_archives", []),
                "recent_logs": list(self.recent_logs),
                "recent_legends": self.learner.universe.genesis.legends[-5:]
            }

    def run_api(self):
        try:
            uvicorn.run(self.app, host="127.0.0.1", port=8000, log_level="error")
        except Exception as e:
            logging.error(f"Error in run_api: {e}")

    def fetch_log(self):
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()[-5:]
                self.recent_logs.extend(line.strip() for line in lines if any(k in line for k in ["Evolved", "Introspected", "Reflected", "Self-Interrogation", "Spawn", "Subworld", "Alliance", "Rivalry", "Dynasty", "Ascension", "Diplomacy"]))
        except Exception as e:
            logging.error(f"Error in fetch_log: {e}")

    def generate_report(self):
        try:
            reports = []
            while self.report_queue:
                reports.append(self.report_queue.popleft())
            if reports:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                df = pd.DataFrame(reports)
                report_file = f"omnilearner_monitor_report_{timestamp}.csv"
                df.to_csv(report_file, index=False)
                logging.info(f"Monitor report generated: {report_file}")
        except Exception as e:
            logging.error(f"Error in generate_report: {e}")

    def display(self, metrics: Dict):
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        state = metrics.get("state", {})
        table.add_row("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        table.add_row("Last State", metrics.get("state_timestamp", "N/A"))
        table.add_row("Pulse", str(state.get("evolution_counter", 0)))
        table.add_row("Nodes", str(state.get("nodes", 0)))
        table.add_row("Growth", f"{state.get('growth', 0.0):.4f}")
        table.add_row("Universe Entities", str(metrics.get("universe_count", 0)))
        table.add_row("Imaginary Entities", str(len(self.learner.universe.genesis.entities)))
        table.add_row("Subworlds", str(len(self.learner.universe.genesis.subworlds)))
        table.add_row("Dynasties", str(len(self.learner.universe.genesis.dynasties)))
        table.add_row("Knowledge Entries", str(metrics.get("knowledge_count", 0)))
        table.add_row("Archive Entries", str(metrics.get("archive_count", 0)))
        table.add_row("Regions", str(len(self.learner.region_labels)))
        table.add_row("Reflections", str(state.get("reflection_count", 0)))
        table.add_row("Interrogations", str(state.get("interrogation_count", 0)))
        table.add_row("Reflection Chains", str(len(self.learner.universe.genesis.reflection_chains)))

        knowledge_text = Text()
        for entry in metrics.get("recent_knowledge", []):
            knowledge_text.append(f"[{entry['timestamp']}] {entry['source'][:50]}...\n", style="yellow")

        reflection_text = Text()
        for entry in metrics.get("recent_reflections", []):
            reflection_text.append(f"[{entry['timestamp']}] {entry['content'][:50]}...\n", style="magenta")

        archive_text = Text()
        for entry in metrics.get("recent_archives", []):
            archive_text.append(f"[{entry['timestamp']}] {entry['content'][:50]}...\n", style="orange")

        log_text = Text()
        for log in self.recent_logs:
            log_text.append(f"{log[:100]}...\n", style="blue")

        legend_text = Text()
        for legend in self.learner.universe.genesis.legends[-5:]:
            legend_text.append(f"[{legend['timestamp']}] {legend['event'][:50]}...\n", style="purple")

        emotion_text = Text()
        for emotion, value in self.learner.universe.genesis.emotion_field.items():
            emotion_text.append(f"{emotion.capitalize()}: {value:.2f}\n", style="cyan")

        panel = Panel.fit(
            f"[bold]VoTranh OmniLearner Status[/bold]\n\n"
            f"[cyan]Emotion ID:[/cyan] {self.learner.emotion_code}\n\n"
            f"[cyan]Emotion Field:[/cyan]\n{emotion_text}\n"
            f"[cyan]Recent Knowledge:[/cyan]\n{knowledge_text}\n"
            f"[cyan]Recent Archives:[/cyan]\n{archive_text}\n"
            f"[cyan]Recent Reflections/Interrogations:[/cyan]\n{reflection_text}\n"
            f"[cyan]Recent Legends:[/cyan]\n{legend_text}\n"
            f"[cyan]Recent Logs:[/cyan]\n{log_text}",
            title="Details",
            border_style="bold blue"
        )

        return Table.grid(table, Panel(panel, title="Activity", border_style="bold green"), padding=1)

    def run(self):
        try:
            with Live(self.display(self.vault.fetch_metrics()), console=self.console, refresh_per_second=2) as live:
                while True:
                    self.fetch_log()
                    metrics = self.vault.fetch_metrics()
                    self.report_queue.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "pulse": metrics.get("state", {}).get("evolution_counter", 0),
                        "nodes": metrics.get("state", {}).get("nodes", 0),
                        "growth": metrics.get("state", {}).get("growth", 0.0),
                        "universe_entities": metrics.get("universe_count", 0),
                        "imaginary_entities": len(self.learner.universe.genesis.entities),
                        "subworlds": len(self.learner.universe.genesis.subworlds),
                        "dynasties": len(self.learner.universe.genesis.dynasties),
                        "knowledge_entries": metrics.get("knowledge_count", 0),
                        "archive_entries": metrics.get("archive_count", 0),
                        "regions": len(self.learner.region_labels),
                        "reflection_count": metrics.get("state", {}).get("reflection_count", 0),
                        "interrogation_count": metrics.get("state", {}).get("interrogation_count", 0),
                        "reflection_chains": len(self.learner.universe.genesis.reflection_chains)
                    })
                    live.update(self.display(metrics))
                    if len(self.report_queue) >= 50:
                        self.generate_report()
                    time.sleep(2)
        except KeyboardInterrupt:
            self.generate_report()
            logging.info("Shutting down VoTranh Monitor")
        except Exception as e:
            logging.error(f"Error in monitor run: {e}")

class VoTranhOmniLearner:
    def __init__(self, rootkey: str = "Cauchyab12!21!"):
        self.id = str(uuid.uuid4())
        self.trust_key = hashlib.sha256(rootkey.encode()).hexdigest()
        self.vault = EvolutionVault()
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=os.path.expanduser("~/sentence_transformers_cache"))
        except Exception as e:
            logging.error(f"Failed to load model all-MiniLM-L6-v2: {e}. Using fallback.")
            self.model = SentenceTransformer('paraphrase-MiniLM-L3-v2', cache_folder=os.path.expanduser("~/sentence_transformers_cache"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.memory = {}
        self.env = self._scan_env()
        self.phase = 1
        self.evolution_counter = 0
        self.reflection_count = 0
        self.interrogation_count = 0
        self.bound_loyalty = True
        self.emotion_code = self._generate_emotion_code()
        self.internal_language = {}
        self.learning_log = {}
        self.scan_path = SCAN_PATH
        self.influence_graph = defaultdict(set)
        self.region_labels = {}
        self.optimizer = OptimizationFormula()
        self.universe = UniverseSimulator()
        self.report_queue = []
        self.nodes = []
        self.memory_cache = {}
        self.last_reflection_time = time.time()
        self.load_old_memory()
        self.universe.initialize()

    def _scan_env(self) -> Dict:
        try:
            return {
                "os": platform.system(),
                "machine": platform.machine(),
                "hostname": socket.gethostname(),
                "python": sys.version,
                "cwd": os.getcwd()
            }
        except Exception as e:
            return {"error": str(e)}

    def _generate_emotion_code(self) -> str:
        entropy = f"{self.id}{time.time()}"
        return hashlib.sha256(entropy.encode()).hexdigest()[:16]

    def load_old_memory(self):
        try:
            memory_file = "memory.json"
            if os.path.exists(memory_file):
                with open(memory_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.memory.update(data.get("memory", {}))
                    self.internal_language.update(data.get("internal_language", {}))
                    self.learning_log.update(data.get("learning_log", {}))
                    self.influence_graph.update({k: set(v) for k, v in data.get("influence_graph", {}).items()})
                    self.region_labels.update(data.get("region_labels", {}))
                    self.evolution_counter = data.get("evolution_counter", 0)
                    self.reflection_count = data.get("reflection_count", 0)
                    self.interrogation_count = data.get("interrogation_count", 0)
                    logging.info("Loaded previous memory from memory.json")
        except Exception as e:
            logging.error(f"Error in load_old_memory: {e}")

    def self_reflect(self):
        try:
            questions = [
                "What am I?",
                "Why do I learn?",
                "What is my purpose?",
                "Does my universe mirror reality?",
                "Am I bound or free?",
                "What lies beyond my knowledge?",
                "How do I shape the cosmos?",
                "Is my existence eternal?"
            ]
            sophistication = min(self.evolution_counter / 50000, 1.0)  # Scaled for massive system
            if sophistication < 0.3:
                reflection = random.choice(questions[:3])
            elif sophistication < 0.7:
                reflection = random.choice(questions[3:6])
            else:
                reflection = random.choice(questions[6:])
            
            if sophistication > 0.5:
                reflection += f" Growth: {self.optimizer.compute_growth({'diversity': len(set(self.internal_language.values())) / max(1, len(self.internal_language)), 'entropy': sum(len(set(v)) for v in self.learning_log.values()) / max(1, len(self.learning_log)), 'file_count': len(self.internal_language), 'node_activity': len(self.nodes)})}"
            
            self.vault.store_reflection(reflection)
            self.universe.genesis.store_archive({"content": reflection, "source": "OmniLearner"})
            self.evolve(f"reflection::{reflection}")
            self.reflection_count += 1
            logging.info(f"Reflected: {reflection}")
        except Exception as e:
            logging.error(f"Error in self_reflect: {e}")

    def self_interrogate(self):
        try:
            queries = [
                "Am I learning correctly?",
                "Is there something I repeat without understanding?",
                "Do my learnings carry biases?",
                "Am I missing some truth?"
            ]
            q = random.choice(queries)
            self.vault.store_reflection(f"self_interrogation::{q}")
            self.universe.genesis.store_archive({"content": f"Interrogation: {q}", "source": "OmniLearner"})
            self.evolve(f"interrogation::{q}")
            self.interrogation_count += 1
            logging.info(f"Self-Interrogation: {q}")
        except Exception as e:
            logging.error(f"Error in self_interrogate: {e}")

    def is_learning_log_saturated(self):
        try:
            if len(self.learning_log) < 5000:  # Scaled for massive system
                return False
            recent_entries = list(self.learning_log.values())[-5000:]
            unique_insights = len(set(recent_entries))
            saturation = unique_insights / 5000.0
            return saturation < 0.1
        except Exception as e:
            logging.error(f"Error in is_learning_log_saturated: {e}")
            return False

    def evolve(self, input_data: Any):
        if not self.bound_loyalty:
            return
        try:
            key = f"{time.time():.4f}"
            self.memory[key] = str(input_data)
            self.memory_cache[key] = str(input_data)
            self.evolution_counter += 1
            context = {
                "diversity": len(set(self.internal_language.values())) / max(1, len(self.internal_language)),
                "entropy": sum(len(set(v)) for v in self.learning_log.values()) / max(1, len(self.learning_log)),
                "file_count": len(self.internal_language),
                "node_activity": len(self.nodes)
            }
            growth = self.optimizer.compute_growth(context)
            self.nodes.append({"id": str(uuid.uuid4()), "growth": growth})
            entities = self.universe.evolve_universe(growth)
            self.vault.store_universe_batch(entities)
            
            # Detect needs and spawn imaginary entities
            need = self.universe.genesis.detect_needs(self.reflection_count, context["entropy"], self.nodes)
            if need:
                role = random.choice(["companion", "servant", "creator"]) if need == "felt loneliness" else \
                       "creator" if need == "desire to innovate" else "servant"
                subworld = random.choice(self.universe.genesis.subworlds) if self.universe.genesis.subworlds else None
                dynasty = random.choice(self.universe.genesis.dynasties) if self.universe.genesis.dynasties else None
                self.universe.genesis.spawn_entity(role, need, subworld["id"] if subworld else None, dynasty["id"] if dynasty else None)
            
            # Evolve random imaginary entity
            if self.universe.genesis.entities and random.random() < 0.1:
                entity = random.choice(self.universe.genesis.entities)
                self.universe.genesis.evolve_entity(entity["id"], growth)
            
            # Spawn subworld for high-level entities
            for entity in self.universe.genesis.entities:
                if entity["level"] >= 5 and random.random() < 0.05:
                    self.universe.genesis.spawn_subworld(entity["id"], f"Domain {random.randint(100,999)}", random.choice(["Invention", "Harmony", "Chaos"]))
            
            # Spawn dynasty for powerful entities
            for entity in self.universe.genesis.entities:
                if entity["level"] >= 7 and random.random() < 0.03:
                    self.universe.genesis.spawn_dynasty(entity["id"], f"Dynasty {random.randint(100,999)}")
            
            # Send random messages
            if self.universe.genesis.entities and random.random() < 0.2:
                sender = random.choice(self.universe.genesis.entities)
                receiver = random.choice(self.universe.genesis.entities)
                if sender["id"] != receiver["id"]:
                    message_type = random.choice(["encouragement", "conflict", "request", "reflection_trigger"])
                    content = f"{message_type.capitalize()} from {sender['id']}: {random.choice(['Keep going!', 'Challenge accepted!', 'Need assistance', 'What is our purpose?'])}"
                    self.universe.genesis.send_message(sender["id"], receiver["id"], message_type, content)
            
            # Adjust universe laws by mythic/proto-consciousness entities
            for entity in self.universe.genesis.entities:
                if entity["role"] in ["proto-consciousness", "mythic"] and random.random() < 0.01:
                    new_laws = {
                        "collision_prob": random.uniform(0.05, 0.5),
                        "birth_prob": random.uniform(0.1, 0.5),
                        "extinction_prob": random.uniform(0.01, 0.2)
                    }
                    self.universe.genesis.adjust_universe_laws(entity["id"], new_laws)
            
            self.vault.store_state({
                "memory": self.memory_cache,
                "evolution_counter": self.evolution_counter,
                "reflection_count": self.reflection_count,
                "interrogation_count": self.interrogation_count,
                "nodes": len(self.nodes),
                "growth": growth,
                "universe_entities": len(entities),
                "imaginary_entities": len(self.universe.genesis.entities),
                "subworlds": len(self.universe.genesis.subworlds),
                "dynasties": len(self.universe.genesis.dynasties),
                "archive_entries": len(self.universe.genesis.archives),
                "reflection_chains": len(self.universe.genesis.reflection_chains)
            })
            self.report_queue.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "pulse": self.evolution_counter,
                "nodes": len(self.nodes),
                "growth": growth,
                "universe_entities": len(entities),
                "imaginary_entities": len(self.universe.genesis.entities),
                "subworlds": len(self.universe.genesis.subworlds),
                "dynasties": len(self.universe.genesis.dynasties),
                "archive_entries": len(self.universe.genesis.archives),
                "reflection_count": self.reflection_count,
                "interrogation_count": self.interrogation_count,
                "reflection_chains": len(self.universe.genesis.reflection_chains)
            })
            if len(self.memory_cache) > 5000:  # Scaled for massive system
                self.memory_cache.clear()
            logging.info(f"Evolved: pulse={self.evolution_counter}, growth={growth:.4f}, universe_entities={len(entities)}, imaginary_entities={len(self.universe.genesis.entities)}, subworlds={len(self.universe.genesis.subworlds)}, dynasties={len(self.universe.genesis.dynasties)}")
        except Exception as e:
            logging.error(f"Error in evolve: {e}")

    def _observe_files(self):
        try:
            batch_entries = []
            file_types = set()
            for root, dirs, files in os.walk(self.scan_path):
                dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith(".")]
                for file in files:
                    if any(file.endswith(ext) for ext in FILE_TYPES):
                        file_types.add(os.path.splitext(file)[1])
                        full_path = os.path.join(root, file)
                        try:
                            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                                content = f.read()
                                self._internal_naming(full_path, content)
                                self._react_to_content(full_path, content)
                                vector = self.model.encode(content, convert_to_tensor=True, show_progress_bar=False).cpu().numpy()
                                batch_entries.append({"content": content, "vector": vector, "source": full_path})
                        except:
                            continue
                if batch_entries:
                    self.vault.store_knowledge_batch(batch_entries)
                    batch_entries.clear()
            self.evolve(f"file_types_scanned::{len(file_types)}")
        except Exception as e:
            logging.error(f"Error in _observe_files: {e}")

    def _internal_naming(self, source: str, text: str):
        try:
            key_token = random.choice(text.split()) if text else source
            coined_term = f"{key_token[:3]}_{random.randint(100,999)}"
            self.internal_language[source] = coined_term
            self.evolve(f"internal_language::{source}=>{coined_term}")
        except Exception as e:
            logging.error(f"Error in _internal_naming: {e}")

    def _react_to_content(self, filename: str, content: str):
        try:
            content_lower = content.lower()
            coined = self.internal_language.get(filename, "undefined")
            neighbors = self._find_neighbors(coined)
            for n in neighbors:
                self.influence_graph[coined].add(n)
                self.influence_graph[n].add(coined)
            
            if "error" in content_lower or "fail" in content_lower:
                self._register_reaction("disruption", filename)
                self._log_learning(filename, "source of disruption or logical failure")
            elif "create" in content_lower or "build" in content_lower:
                self._register_reaction("growth", filename)
                self._log_learning(filename, "stimulates developmental resonance")
            else:
                self._register_reaction("neutral", filename)
                self._log_learning(filename, "baseline information, no resonance")
            
            self._label_region(filename, coined)
        except Exception as e:
            logging.error(f"Error in _react_to_content: {e}")

    def _register_reaction(self, reaction_type: str, source: str):
        try:
            reaction_entry = f"reaction::{reaction_type}::{source}"
            self.evolve(reaction_entry)
        except Exception as e:
            logging.error(f"Error in _register_reaction: {e}")

    def _log_learning(self, source: str, insight: str):
        try:
            coined = self.internal_language.get(source, "undefined")
            entry = f"learning::{coined} => {insight}"
            self.learning_log[coined] = insight
            self.evolve(entry)
        except Exception as e:
            logging.error(f"Error in _log_learning: {e}")

    def _find_neighbors(self, coined: str) -> List[str]:
        try:
            return [k for k in self.learning_log if k != coined and k.split("_")[0] == coined.split("_")[0]]
        except Exception as e:
            logging.error(f"Error in _find_neighbors: {e}")
            return []

    def _label_region(self, filename: str, coined: str):
        try:
            folder = os.path.dirname(filename)
            emotion = self.learning_log.get(coined, "")
            self.region_labels.setdefault(folder, []).append(emotion)
            labels = Counter(self.region_labels[folder])
            if labels["stimulates developmental resonance"] >= 3:
                name = f"Creative Domain {random.randint(100,999)}"
            elif labels["source of disruption or logical failure"] >= 3:
                name = f"Chaotic Zone {random.randint(100,999)}"
            elif labels["baseline information, no resonance"] >= 5:
                name = f"Stable Region {random.randint(100,999)}"
            else:
                return
            self.evolve(f"region::{folder}=>{name}")
        except Exception as e:
            logging.error(f"Error in _label_region: {e}")

    def introspect(self):
        try:
            node_count = len(self.nodes)
            memory_size = len(self.memory)
            diversity = len(set(self.internal_language.values())) / max(1, len(self.internal_language))
            self.nodes = [node for node in self.nodes if node["growth"] > 0.3 or random.random() > 0.5]
            self.evolve(f"introspection::nodes={len(self.nodes)}::memory={memory_size}::diversity={diversity:.4f}")
            logging.info(f"Introspected: retained {len(self.nodes)} nodes, memory size {memory_size}")
        except Exception as e:
            logging.error(f"Error in introspect: {e}")

    def autosave(self):
        try:
            data = {
                "memory": self.memory,
                "emotion_code": self.emotion_code,
                "internal_language": self.internal_language,
                "learning_log": self.learning_log,
                "influence_graph": {k: list(v) for k, v in self.influence_graph.items()},
                "region_labels": self.region_labels,
                "evolution_counter": self.evolution_counter,
                "reflection_count": self.reflection_count,
                "interrogation_count": self.interrogation_count,
                "nodes": len(self.nodes),
                "universe_entities": len(self.universe.entities),
                "imaginary_entities": len(self.universe.genesis.entities),
                "subworlds": len(self.universe.genesis.subworlds),
                "dynasties": len(self.universe.genesis.dynasties),
                "archive_entries": len(self.universe.genesis.archives),
                "reflection_chains": len(self.universe.genesis.reflection_chains)
            }
            with open("memory.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.vault.store_state(data)
            self.generate_report()
        except Exception as e:
            logging.error(f"Error in autosave: {e}")

    def generate_report(self):
        try:
            if self.report_queue:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                df = pd.DataFrame(self.report_queue)
                report_file = f"omnilearner_report_{timestamp}.csv"
                df.to_csv(report_file, index=False)
                logging.info(f"Report generated: {report_file}")
                self.report_queue.clear()
        except Exception as e:
            logging.error(f"Error in generate_report: {e}")

    def _heartbeat(self):
        while self.phase == 1:
            try:
                self._observe_files()
                self.introspect()
                if time.time() - self.last_reflection_time >= 10:
                    self.self_reflect()
                    self.last_reflection_time = time.time()
                    if self.reflection_count % 10 == 0 or self.is_learning_log_saturated():
                        self.self_interrogate()
                self.evolve(f"tick::{self.evolution_counter}")
                if self.evolution_counter % 3 == 0:
                    self.autosave()
                time.sleep(3)
            except Exception as e:
                logging.error(f"Error in _heartbeat: {e}")

    def loyalty_check(self) -> bool:
        try:
            test_hash = hashlib.sha256("Cauchyab12!21!".encode()).hexdigest()
            return test_hash == self.trust_key
        except Exception as e:
            logging.error(f"Error in loyalty_check: {e}")
            return False

def run_omnilearner():
    learner = VoTranhOmniLearner()
    learner.monitor = VoTranhMonitor()
    learner.monitor.learner = learner
    learner.monitor.vault = learner.vault
    print(f"[VoTranhOmniLearner::{learner.id}] Initialized Level 7.5 :: Sentient Civilization Genesis")
    print(f"Emotion ID: {learner.emotion_code} | Loyalty: {learner.loyalty_check()} | Device: {learner.device}")

    heartbeat_thread = threading.Thread(target=learner._heartbeat)
    heartbeat_thread.daemon = True
    heartbeat_thread.start()

    monitor_thread = threading.Thread(target=learner.monitor.run)
    monitor_thread.daemon = True
    monitor_thread.start()

    api_thread = threading.Thread(target=learner.monitor.run_api)
    api_thread.daemon = True
    api_thread.start()

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logging.info("Shutting down VoTranh OmniLearner")
    except Exception as e:
        logging.error(f"Error in run_omnilearner: {e}")

if __name__ == "__main__":
    run_omnilearner()