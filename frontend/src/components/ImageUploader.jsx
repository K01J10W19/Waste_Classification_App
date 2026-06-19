import React, { useRef } from "react";

export default function ImageUploader({ onUpload, loading }) {
  const inputRef = useRef(null);

  const handleChange = (e) => {
    const file = e.target.files[0];
    if (file) onUpload(file);
  };

  return (
    <div
      className="border-2 border-dashed border-green-400 rounded-xl p-10 text-center cursor-pointer hover:bg-green-50 transition"
      onClick={() => inputRef.current.click()}
    >
      <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={handleChange} />
      {loading ? (
        <p className="text-green-600 font-medium">Classifying...</p>
      ) : (
        <p className="text-gray-500">Click or drag a waste image here to classify</p>
      )}
    </div>
  );
}
