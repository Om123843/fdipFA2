import React from "react"

export default function ResultCard({ prediction, confidence }) {
  const isSewage = prediction && prediction.toLowerCase().includes("sewage")
  return (
    <div className="mt-2 p-4 rounded border">
      <div className={`text-xl font-bold ${isSewage ? "text-red-600" : "text-green-600"}`}>Result: {prediction}</div>
      <div className="mt-2 text-sm text-slate-600">Confidence: {(confidence * 100).toFixed(1)}%</div>
    </div>
  )
}
