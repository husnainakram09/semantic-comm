import { useState, useCallback } from 'react'
import {
  encodeImage,
  transmitThroughChannel,
  decodeImage,
  calculateMetrics,
} from '../services/api'

/**
 * Custom hook for managing the semantic communication pipeline
 * @returns {Object} Pipeline state and control functions
 */
export function usePipeline() {
  const [stage, setStage] = useState('idle') // idle, encoding, transmitting, decoding, done, error
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)
  const [caption, setCaption] = useState(null)
  const [progress, setProgress] = useState(0)

  /**
   * Run the complete pipeline
   * @param {File} imageFile - The image file to process
   * @param {number} snrDb - Signal-to-Noise Ratio in dB
   * @param {string} channelType - Type of channel ('awgn', 'rayleigh', 'none')
   */
  const runPipeline = useCallback(async (imageFile, snrDb, channelType = 'awgn') => {
    if (!imageFile) {
      setError('No image file provided')
      setStage('error')
      return
    }

    setError(null)
    setResults(null)
    setCaption(null)
    setProgress(0)

    try {
      // Step 1: Encode
      setStage('encoding')
      setProgress(25)
      const formData = new FormData()
      formData.append('image', imageFile)
      const encodeRes = await encodeImage(formData)
      const { latent_code, caption: extractedCaption } = encodeRes
      setCaption(extractedCaption)

      // Step 2: Transmit
      setStage('transmitting')
      setProgress(50)
      const transmitRes = await transmitThroughChannel(latent_code, channelType, snrDb)
      const { noisy_latent } = transmitRes

      // Step 3: Decode
      setStage('decoding')
      setProgress(75)
      const decodeRes = await decodeImage(noisy_latent)
      const { reconstructed_image } = decodeRes

      // Step 4: Calculate Metrics
      setProgress(90)
      const originalImageUrl = URL.createObjectURL(imageFile)
      const metricsRes = await calculateMetrics(originalImageUrl, reconstructed_image)

      setResults({
        reconstructed: reconstructed_image,
        metrics: metricsRes,
      })
      setStage('done')
      setProgress(100)
    } catch (err) {
      console.error('Pipeline error:', err)
      const errorMsg = 
        err.response?.data?.error ||
        err.response?.data?.message ||
        err.message ||
        'Pipeline failed. Please try again.'
      setError(errorMsg)
      setStage('error')
      setProgress(0)
    }
  }, [])

  /**
   * Reset the pipeline to idle state
   */
  const reset = useCallback(() => {
    setStage('idle')
    setResults(null)
    setError(null)
    setCaption(null)
    setProgress(0)
  }, [])

  /**
   * Check if pipeline is currently processing
   */
  const isProcessing = ['encoding', 'transmitting', 'decoding'].includes(stage)

  return {
    stage,
    results,
    error,
    caption,
    progress,
    isProcessing,
    runPipeline,
    reset,
  }
}

export default usePipeline
