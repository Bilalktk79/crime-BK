import os
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Auto-create spam_detector_model.pkl if missing
if not os.path.exists("spam_detector_model.pkl"):
    print("Creating dummy spam_detector_model.pkl ...")
    X = ["help", "attack", "emergency", "normal", "sale", "discount"]
    y = [1, 1, 1, 0, 0, 0]
    vectorizer = CountVectorizer()
    X_vect = vectorizer.fit_transform(X)
    model = MultinomialNB()
    model.fit(X_vect, y)
    joblib.dump(model, "spam_detector_model.pkl")
    joblib.dump(vectorizer, "spam_vectorizer.pkl")

# Auto-create incident_classifier.pkl if missing
if not os.path.exists("incident_classifier.pkl"):
    print("🔧 Creating dummy incident_classifier.pkl ...")
    X = ["fire", "robbery", "theft", "accident"]
    y = ["fire", "crime", "crime", "accident"]
    vectorizer2 = CountVectorizer()
    X_vect2 = vectorizer2.fit_transform(X)
    model2 = MultinomialNB()
    model2.fit(X_vect2, y)
    joblib.dump(model2, "incident_classifier.pkl")
# ─── Imports ─────────────────────────────────────────────────────────
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
from pymongo import MongoClient
from datetime import datetime
from bson.objectid import ObjectId

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import re

spam_model = joblib.load("spam_detector_model.pkl")
spam_vectorizer = joblib.load("spam_vectorizer.pkl")

# ─── App & DB Setup ──────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

client = MongoClient("mongodb://localhost:27017/")
db = client["crime_report_db"]
collection = db["incidents"]

# ─── Load ML Models ─────────────────────────────────────────
try:
    model = joblib.load("incident_classifier.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except Exception as e:
    print(f"Incident model loading error: {e}")
    model = None
    vectorizer = None

try:
    spam_model = joblib.load("spam_detector_model.pkl")
    spam_vectorizer = joblib.load("spam_vectorizer.pkl")
except Exception as e:
    print(f"Spam detection model load failed: {e}")
    spam_model = None
    spam_vectorizer = None

# ─── Spam & Suspicion Detection Logic ────────────────────────────────

def is_suspicious(text):
    if not text or len(text.strip()) < 10:
        return True
    if any(word in text.lower() for word in ["asdf", "qwer", "lorem", "test", "dummy"]):
        return True
    if re.search(r"(.)\1{4,}", text):
        return True
    if len(set(text.lower().split())) < 3:
        return True
    try:
        if spam_model is not None and spam_vectorizer is not None:
            desc_vector = spam_vectorizer.transform([text])
            is_spam = spam_model.predict(desc_vector)[0]
            return bool(is_spam)
        else:
            return False
    except Exception as e:
        print(f"Spam detection predict error: {e}")
        return False

# ─── Routes ────────────────────────────────────────────
@app.route("/")
def index():
    return jsonify({"status": "server is running"}), 200

@app.route("/check_spam", methods=["POST"])
def check_spam():
    try:
        description = request.json.get("description", "")
        if not description.strip():
            return jsonify({"error": "Description is empty"}), 400

        if spam_model is None or spam_vectorizer is None:
            return jsonify({"error": "Spam detection models not loaded"}), 500

        desc_vector = spam_vectorizer.transform([description])
        is_spam = spam_model.predict(desc_vector)[0]

        return jsonify({"is_spam": bool(is_spam)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict-type", methods=["POST"])
def predict_type():
    try:
        description = request.json.get("description", "")
        if not description.strip():
            return jsonify({"error": "Description missing"}), 400

        if model is None or vectorizer is None:
            return jsonify({"error": "Incident classification models not loaded"}), 500

        vector = vectorizer.transform([description])
        prediction = model.predict(vector)[0]
        return jsonify({"predicted_type": prediction}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/report", methods=["POST"])
def report_incident():
    print("Received report request")
    try:
        data = request.get_json()
        print("Data:", data)
        required_fields = ["location", "severity", "description"]
        if not all(data.get(field) for field in required_fields):
            return jsonify({"status": "error", "message": "Missing fields"}), 400

        description = data["description"]

        # Spam detection
        if spam_model is not None and spam_vectorizer is not None:
            vector = spam_vectorizer.transform([description])
            is_spam = spam_model.predict(vector)[0]
        else:
            is_spam = False

        # Type classification
        if model is None or vectorizer is None:
            predicted_type = "unknown"
        else:
            predicted_type = model.predict(vectorizer.transform([description]))[0]

        # Latitude/Longitude parsing
        try:
            latitude = float(data.get("latitude", 0))
            longitude = float(data.get("longitude", 0))
        except (ValueError, TypeError):
            latitude = 0.0
            longitude = 0.0

        now = datetime.utcnow()
        incident = {
            "type": predicted_type,
            "location": data["location"],
            "severity": data["severity"],
            "description": description,
            "latitude": latitude,
            "longitude": longitude,
            "timestamp": now,
            "approved": False,
            "flagged": is_suspicious(description),
            "spam": bool(is_spam),
        }

        inserted = collection.insert_one(incident)
        incident["_id"] = str(inserted.inserted_id)
        incident["timestamp"] = now.isoformat()

        # ✅ Only emit if not spam
        if not is_spam:
            socketio.emit("new_incident", incident)

        return jsonify({"status": "success", "data": incident, "spam": bool(is_spam)}), 201

    except ValueError:
        return jsonify({"status": "error", "message": "Invalid latitude/longitude"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# (rest of the routes remain unchanged)


@app.route("/incidents", methods=["GET"])
def get_incidents():
    try:
        query = {"approved": True, "flagged": False, "spam": False}
        incidents = [
            {**doc, "_id": str(doc["_id"]), "timestamp": doc["timestamp"].isoformat()}
            for doc in collection.find(query).sort("timestamp", -1)
        ]
        return jsonify({"status": "success", "data": incidents}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ── Admin Routes ──────────────────────────────────────────────────────────

@app.route("/admin/incidents", methods=["GET"])
def get_admin_incidents():
    try:
        incidents = [
            {
                **doc,
                "_id": str(doc["_id"]),
                "timestamp": doc["timestamp"].isoformat(),
                "flagged": doc.get("flagged", False),
                "approved": doc.get("approved", False),
            }
            for doc in collection.find().sort("timestamp", -1)
        ]
        return jsonify({"status": "success", "data": incidents}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/admin/incidents/<incident_id>/approve", methods=["POST"])
def approve_incident(incident_id):
    try:
        result = collection.update_one(
            {"_id": ObjectId(incident_id)},
            {"$set": {"approved": True, "flagged": False}}
        )
        if result.matched_count == 0:
            return jsonify({"status": "error", "message": "Incident not found"}), 404

        doc = collection.find_one({"_id": ObjectId(incident_id)})
        doc["_id"] = str(doc["_id"])
        doc["timestamp"] = doc["timestamp"].isoformat()
        socketio.emit("new_incident", doc)

        return jsonify({"status": "success", "message": "Incident approved"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/admin/incidents/<incident_id>/remove", methods=["DELETE"])
def remove_incident(incident_id):
    try:
        result = collection.delete_one({"_id": ObjectId(incident_id)})
        if result.deleted_count == 0:
            return jsonify({"status": "error", "message": "Incident not found"}), 404
        return jsonify({"status": "success", "message": "Incident removed"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/admin/flag", methods=["POST"])
def flag_incident():
    try:
        data = request.json
        incident_id = data.get("id")
        flagged = data.get("flagged")

        if not incident_id or flagged is None:
            return jsonify({"status": "error", "message": "Missing id or flag"}), 400

        result = collection.update_one({"_id": ObjectId(incident_id)}, {"$set": {"flagged": flagged}})
        if result.matched_count == 0:
            return jsonify({"status": "error", "message": "Incident not found"}), 404

        return jsonify({"status": "success", "message": "Flag updated", "flagged": flagged}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/admin/incidents/flagged", methods=["GET"])
def get_flagged_reports():
    try:
        flagged = [
            {
                **doc,
                "_id": str(doc["_id"]),
                "timestamp": doc["timestamp"].isoformat()
            }
            for doc in collection.find({"flagged": True}).sort("timestamp", -1)
        ]
        return jsonify({"status": "success", "data": flagged}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/remove_report/<report_id>", methods=["DELETE"])
def remove_report(report_id):
    try:
        result = collection.delete_one({"_id": ObjectId(report_id)})
        if result.deleted_count == 0:
            return jsonify({"status": "error", "message": "Report not found"}), 404
        return jsonify({"status": "success", "message": "Report removed"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/crime-hotspots", methods=["GET"])
def crime_hotspots():
    try:
        query = {"latitude": {"$ne": None}, "longitude": {"$ne": None}, "approved": True, "flagged": False}
        incidents = list(collection.find(query, {"latitude": 1, "longitude": 1, "_id": 0}))

        if len(incidents) < 3:
            return jsonify({"status": "error", "message": "Not enough data for clustering"}), 400

        df = pd.DataFrame(incidents)
        kmeans = KMeans(n_clusters=3, n_init=10)
        kmeans.fit(df)
        centers = kmeans.cluster_centers_.tolist()

        return jsonify({"status": "success", "hotspots": centers}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/hotspots", methods=["GET"])
def get_hotspots():
    # Fetch incidents with lat/lng
    incidents = list(collection.find({
        "latitude": {"$exists": True}, 
        "longitude": {"$exists": True}
    }))

    if not incidents:
        return jsonify([])

    coords = np.array([[float(inc["latitude"]), float(inc["longitude"]) ] for inc in incidents])

    # K-Means clustering
    k = min(5, len(coords))  # max 5 clusters or total incidents
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(coords)
    centroids = kmeans.cluster_centers_

    # Build hotspot data
    hotspots = []
    for i in range(k):
        count = np.sum(labels == i)
        hotspots.append({
            "lat": float(centroids[i][0]),
            "lng": float(centroids[i][1]),
            "count": int(count)
        })

    return jsonify(hotspots)

# ─── Run Server ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)