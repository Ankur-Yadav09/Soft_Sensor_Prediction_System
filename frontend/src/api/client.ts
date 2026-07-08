import axios from 'axios'

// Proxied to the FastAPI backend by vite.config.ts's dev-server proxy.
export const apiClient = axios.create({ baseURL: '/api' })
