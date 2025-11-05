import React, { useState } from "react"
import axios from "axios"
import ResultCard from "./ResultCard"

export default function UploadForm() {
  const [image, setImage] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)

  function handleFile(e) {
    setImage(e.target.files[0])
  }

  async function handleSubmit(e) {
    e.preventDefault()
    if (!image) {
      alert("Please upload an image of the water sample.")
      return
    }

    const fd = new FormData()
    fd.append("image", image)
    // Send default sensor readings (image features are primary)
    fd.append("readings", JSON.stringify({ pH: "7.0", turbidity: "5", conductivity: "300", DO: "6", temperature: "22" }))

    try {
      setLoading(true)
      setResult(null)
      const resp = await axios.post("http://localhost:5000/predict", fd, {
        headers: { "Content-Type": "multipart/form-data" },
      })
      setResult(resp.data)
    } catch (err) {
      console.error(err)
      const message = err?.response?.data?.error || err.message || "Request failed"
      setResult({ error: message })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-slate-700">Upload Water Sample Image</label>
          <input type="file" accept="image/*" onChange={handleFile} className="mt-2 w-full" />
          <p className="text-xs text-slate-500 mt-1">Upload an image of the water sample to analyze</p>
        </div>

        <div className="flex items-center gap-3">
          <button type="submit" className="bg-sky-600 text-white px-4 py-2 rounded shadow hover:bg-sky-700">
            Analyze Water Quality
          </button>

          {loading && <div className="spinner" aria-hidden="true"></div>}
        </div>
      </form>

      <div>
        <div className="bg-gradient-to-b from-white to-slate-50 p-4 rounded-lg h-full">
          <h3 className="text-lg font-semibold text-slate-700">Preview & Result</h3>
          <div className="mt-3">
            {image ? (
              <img src={URL.createObjectURL(image)} alt="preview" className="w-full rounded" />
            ) : (
              <div className="h-40 rounded bg-slate-100 flex items-center justify-center text-slate-400">Image preview</div>
            )}
          </div>

          <div className="mt-4">
            {result ? (
              result.error ? (
                <div className="text-red-600">Error: {result.error}</div>
              ) : (
                <ResultCard prediction={result.prediction} confidence={result.confidence} />
              )
            ) : (
              <div className="text-slate-500">No analysis yet — upload a water sample image and click Analyze.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
