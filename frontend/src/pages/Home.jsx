import React from "react"
import UploadForm from "../components/UploadForm"

export default function Home() {
  return (
    <div className="bg-white/60 backdrop-blur-md rounded-2xl shadow-lg p-8">
      <header className="mb-6">
        <h1 className="text-3xl font-bold text-sky-700">AquaVision</h1>
        <p className="mt-2 text-slate-700">Detect sewage contamination using image processing + water quality parameters.</p>
      </header>

      <main>
        <UploadForm />
      </main>
    </div>
  )
}
