'use client'

import { motion } from 'framer-motion'
import { useEffect, useState } from 'react'

export default function LoadingScreen() {
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    const timer = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(timer)
          return 100
        }
        return prev + Math.random() * 15
      })
    }, 100)

    return () => clearInterval(timer)
  }, [])

  return (
    <motion.div
      initial={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-dark-950"
    >
      {/* Background Animation */}
      <div className="absolute inset-0 cyber-grid opacity-10" />
      
      {/* Animated Gradient Background */}
      <div className="absolute inset-0 bg-gradient-to-br from-primary-950/30 via-dark-950 to-cyber-950/30" />
      
      {/* Loading Content */}
      <div className="relative z-10 text-center">
        {/* Logo Animation */}
        <motion.div
          initial={{ scale: 0.5, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="mb-8"
        >
          <div className="relative">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
              className="w-24 h-24 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-primary-500 to-cyber-500 flex items-center justify-center"
            >
              <svg
                className="w-12 h-12 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M13 10V3L4 14h7v7l9-11h-7z"
                />
              </svg>
            </motion.div>
            
            {/* Glow Effect */}
            <motion.div
              animate={{ scale: [1, 1.2, 1], opacity: [0.5, 0.8, 0.5] }}
              transition={{ duration: 2, repeat: Infinity }}
              className="absolute inset-0 w-24 h-24 mx-auto rounded-2xl bg-gradient-to-br from-primary-500 to-cyber-500 blur-xl"
            />
          </div>
          
          <motion.h1
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.3, duration: 0.8 }}
            className="text-4xl font-display font-bold text-gradient mb-2"
          >
            SmartVehicle
          </motion.h1>
          
          <motion.p
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.5, duration: 0.8 }}
            className="text-lg text-gray-400 font-medium"
          >
            Autonomous Mobility Intelligence
          </motion.p>
        </motion.div>

        {/* Progress Bar */}
        <motion.div
          initial={{ width: 0, opacity: 0 }}
          animate={{ width: '300px', opacity: 1 }}
          transition={{ delay: 0.8, duration: 0.8 }}
          className="mx-auto mb-4 h-1 bg-gray-800 rounded-full overflow-hidden"
        >
          <motion.div
            className="h-full bg-gradient-to-r from-primary-500 to-cyber-500 rounded-full"
            style={{ width: `${progress}%` }}
            transition={{ duration: 0.3 }}
          />
        </motion.div>

        {/* Loading Text */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1, duration: 0.8 }}
          className="text-sm text-gray-500 font-mono"
        >
          <motion.span
            animate={{ opacity: [1, 0.5, 1] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          >
            Initializing AI Systems...
          </motion.span>
        </motion.div>

        {/* Floating Particles */}
        {[...Array(6)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-2 h-2 bg-primary-500 rounded-full"
            style={{
              left: `${20 + i * 10}%`,
              top: `${30 + (i % 2) * 40}%`,
            }}
            animate={{
              y: [-20, 20, -20],
              opacity: [0.3, 1, 0.3],
            }}
            transition={{
              duration: 3 + i * 0.5,
              repeat: Infinity,
              delay: i * 0.2,
            }}
          />
        ))}
      </div>
    </motion.div>
  )
}