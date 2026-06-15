import axios from 'axios'

export const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 60000, // 60 seconds for long-running operations
})

/**
 * Encode an image to extract semantic features
 * @param {FormData} formData - FormData containing 'image' field
 * @returns {Promise<{latent_code, caption}>}
 */
export async function encodeImage(formData) {
  const response = await api.post('/encode', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

/**
 * Transmit the latent code through a noisy channel
 * @param {Array} latentCode - The latent representation
 * @param {string} channelType - Channel type ('awgn', 'rayleigh', or 'none')
 * @param {number} snrDb - Signal-to-Noise Ratio in decibels
 * @returns {Promise<{noisy_latent}>}
 */
export async function transmitThroughChannel(latentCode, channelType, snrDb) {
  const response = await api.post('/transmit', {
    latent_code: latentCode,
    channel_type: channelType,
    snr_db: snrDb,
  })
  return response.data
}

/**
 * Decode the noisy latent code back to an image
 * @param {Array} noisyLatent - The noisy latent representation
 * @returns {Promise<{reconstructed_image}>}
 */
export async function decodeImage(noisyLatent) {
  const response = await api.post('/decode', {
    noisy_latent: noisyLatent,
  })
  return response.data
}

/**
 * Calculate quality metrics comparing original and reconstructed images
 * @param {string} originalImage - Base64 or URL of original image
 * @param {string} reconstructedImage - Base64 or URL of reconstructed image
 * @returns {Promise<{psnr, ssim, bleu, channel_info}>}
 */
export async function calculateMetrics(originalImage, reconstructedImage) {
  const response = await api.post('/metrics', {
    original_image: originalImage,
    reconstructed_image: reconstructedImage,
  })
  return response.data
}

/**
 * Run the complete pipeline in one call
 * @param {FormData} formData - FormData containing 'image' field
 * @param {string} channelType - Channel type
 * @param {number} snrDb - Signal-to-Noise Ratio
 * @returns {Promise<{latent_code, caption, noisy_latent, reconstructed_image, metrics}>}
 */
export async function runFullPipeline(formData, channelType, snrDb) {
  const response = await api.post('/pipeline', formData, {
    params: {
      channel_type: channelType,
      snr_db: snrDb,
    },
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

export default api
