'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  Activity, 
  AlertTriangle, 
  Eye, 
  Shield, 
  Zap, 
  Car,
  Navigation,
  Gauge,
  Camera,
  Radar
} from 'lucide-react'

export default function VehicleDashboard() {
  const [activeAlert, setActiveAlert] = useState(0)
  const [speed, setSpeed] = useState(45)
  const [detections, setDetections] = useState(0)

  const alerts = [
    { type: 'collision', message: 'Collision Risk Detected', severity: 'high' },
    { type: 'pedestrian', message: 'Pedestrian Detected', severity: 'medium' },
    { type: 'lane', message: 'Lane Departure Warning', severity: 'low' },
    { type: 'fatigue', message: 'Driver Fatigue Alert', severity: 'high' },
  ]

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveAlert((prev) => (prev + 1) % alerts.length)
      setSpeed(Math.floor(Math.random() * 30) + 35)
      setDetections((prev) => prev + Math.floor(Math.random() * 3))
    }, 3000)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="relative w-full max-w-4xl mx-auto">
      {/* Main Dashboard Container */}
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 1, delay: 0.5 }}
        className="relative bg-gradient-to-br from-gray-900/90 to-black/90 backdrop-blur-xl rounded-3xl border border-cyan-500/30 p-8 shadow-2xl"
      >
        {/* Glowing Border Effect */}
        <div className="absolute inset-0 rounded-3xl bg-gradient-to-r from-cyan-500/20 via-blue-500/20 to-purple-500/20 blur-xl" />
        
        {/* Dashboard Header */}
        <div className="relative z-10 flex items-center justify-between mb-8">
          <div className="flex items-center space-x-4">
            <div className="w-12 h-12 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center">
              <Car className="w-6 h-6 text-white" />
            </div>
            <div>
              <h3 className="text-xl font-bold text-white">SIGHTLINE Dashboard</h3>
              <p className="text-cyan-400 text-sm">Real-time AI Monitoring</p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" />
            <span className="text-green-400 text-sm font-medium">ACTIVE</span>
          </div>
        </div>

        {/* Main Dashboard Grid */}
        <div className="relative z-10 grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Speed & Status */}
          <motion.div
            animate={{ scale: [1, 1.02, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="bg-gradient-to-br from-gray-800/50 to-gray-900/50 rounded-2xl p-6 border border-gray-700/50"
          >
            <div className="flex items-center justify-between mb-4">
              <Gauge className="w-6 h-6 text-cyan-400" />
              <span className="text-xs text-gray-400">SPEED</span>
            </div>
            <div className="text-3xl font-bold text-white mb-2">{speed}</div>
            <div className="text-sm text-gray-400">km/h</div>
            <div className="mt-4 h-2 bg-gray-700 rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-gradient-to-r from-cyan-500 to-blue-500"
                animate={{ width: `${(speed / 80) * 100}%` }}
                transition={{ duration: 0.5 }}
              />
            </div>
          </motion.div>

          {/* Active Alerts */}
          <motion.div
            key={activeAlert}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-gradient-to-br from-red-900/30 to-orange-900/30 rounded-2xl p-6 border border-red-500/30"
          >
            <div className="flex items-center justify-between mb-4">
              <AlertTriangle className="w-6 h-6 text-red-400" />
              <span className="text-xs text-red-400">ALERT</span>
            </div>
            <div className="text-sm font-medium text-white mb-2">
              {alerts[activeAlert].message}
            </div>
            <div className={`text-xs px-2 py-1 rounded-full inline-block ${
              alerts[activeAlert].severity === 'high' 
                ? 'bg-red-500/20 text-red-400' 
                : alerts[activeAlert].severity === 'medium'
                ? 'bg-yellow-500/20 text-yellow-400'
                : 'bg-blue-500/20 text-blue-400'
            }`}>
              {alerts[activeAlert].severity.toUpperCase()}
            </div>
          </motion.div>

          {/* Detection Counter */}
          <motion.div
            animate={{ scale: [1, 1.05, 1] }}
            transition={{ duration: 1.5, repeat: Infinity }}
            className="bg-gradient-to-br from-green-900/30 to-emerald-900/30 rounded-2xl p-6 border border-green-500/30"
          >
            <div className="flex items-center justify-between mb-4">
              <Eye className="w-6 h-6 text-green-400" />
              <span className="text-xs text-green-400">DETECTIONS</span>
            </div>
            <div className="text-3xl font-bold text-white mb-2">{detections}</div>
            <div className="text-sm text-gray-400">Objects tracked</div>
          </motion.div>
        </div>

        {/* AI Systems Status */}
        <div className="relative z-10 mt-8 grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { icon: Camera, name: 'Vision AI', status: 'active' },
            { icon: Radar, name: 'LiDAR', status: 'active' },
            { icon: Shield, name: 'Safety', status: 'active' },
            { icon: Navigation, name: 'Navigation', status: 'active' },
          ].map((system, index) => (
            <motion.div
              key={system.name}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 + 1 }}
              className="bg-gradient-to-br from-gray-800/30 to-gray-900/30 rounded-xl p-4 border border-gray-700/30"
            >
              <div className="flex items-center space-x-3">
                <system.icon className="w-5 h-5 text-cyan-400" />
                <div>
                  <div className="text-sm font-medium text-white">{system.name}</div>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                    <span className="text-xs text-green-400">ONLINE</span>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Scanning Animation */}
        <motion.div
          className="absolute inset-0 rounded-3xl border-2 border-cyan-500/30"
          animate={{
            boxShadow: [
              '0 0 20px rgba(6, 182, 212, 0.3)',
              '0 0 40px rgba(6, 182, 212, 0.5)',
              '0 0 20px rgba(6, 182, 212, 0.3)',
            ],
          }}
          transition={{ duration: 2, repeat: Infinity }}
        />
      </motion.div>

      {/* Floating Particles */}
      <div className="absolute inset-0 overflow-hidden rounded-3xl">
        {[...Array(20)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-cyan-400 rounded-full"
            animate={{
              x: [Math.random() * 400, Math.random() * 400],
              y: [Math.random() * 300, Math.random() * 300],
              opacity: [0, 1, 0],
            }}
            transition={{
              duration: Math.random() * 3 + 2,
              repeat: Infinity,
              delay: Math.random() * 2,
            }}
          />
        ))}
      </div>
    </div>
  )
}