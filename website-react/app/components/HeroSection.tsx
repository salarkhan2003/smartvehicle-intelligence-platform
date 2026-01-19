'use client'

import { motion } from 'framer-motion'
import { useEffect, useState } from 'react'
import { Shield, Zap, Brain, Eye } from 'lucide-react'
import CountUp from 'react-countup'
import VehicleDashboard from './VehicleDashboard'

export default function HeroSection() {
  const [currentText, setCurrentText] = useState(0)

  const typedTexts = [
    'Vision Intelligence',
    'Smart Detection',
    'Real-time Monitoring',
    'AI-Powered Safety'
  ]

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentText((prev) => (prev + 1) % typedTexts.length)
    }, 3000)

    return () => clearInterval(interval)
  }, [])

  const stats = [
    { number: 99.7, suffix: '%', label: 'Detection Accuracy' },
    { number: 24, suffix: '/7', label: 'Monitoring' },
  ]

  const features = [
    { icon: Eye, label: 'Computer Vision' },
    { icon: Brain, label: 'Neural Networks' },
    { icon: Shield, label: 'Safety Systems' },
    { icon: Zap, label: 'Real-time Processing' },
  ]

  return (
    <section className="relative min-h-screen flex flex-col overflow-hidden">
      {/* Video Background */}
      <div className="absolute inset-0 z-0">
        <video
          autoPlay
          muted
          loop
          playsInline
          className="w-full h-full object-cover"
        >
          <source src="/assets/304445.mp4" type="video/mp4" />
        </video>
        
        {/* Video Overlay */}
        <div className="absolute inset-0 bg-gradient-to-br from-black/70 via-black/50 to-black/70" />
        <div className="absolute inset-0 bg-gradient-to-t from-black via-transparent to-transparent" />
      </div>

      {/* Animated Particles */}
      <div className="absolute inset-0 z-10">
        {[...Array(30)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-cyan-400 rounded-full"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
            }}
            animate={{
              y: [-20, 20, -20],
              opacity: [0.2, 1, 0.2],
              scale: [0.5, 1, 0.5],
            }}
            transition={{
              duration: 3 + Math.random() * 2,
              repeat: Infinity,
              delay: Math.random() * 2,
            }}
          />
        ))}
      </div>

      {/* Content */}
      <div className="relative z-20 container-custom flex-1 flex flex-col">
        {/* Top Section - Navigation Space */}
        <div className="h-20"></div>
        
        {/* Main Content Area */}
        <div className="flex-1 flex flex-col justify-center max-w-7xl mx-auto">
          {/* Vehicle Dashboard */}
          <motion.div
            initial={{ opacity: 0, y: -50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 1 }}
            className="mb-16"
          >
            <VehicleDashboard />
          </motion.div>

          {/* Main Heading and Content */}
          <div className="text-center">
            {/* Badge */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8 }}
              className="inline-flex items-center px-6 py-3 mb-8 bg-white/10 backdrop-blur-sm border border-cyan-500/30 rounded-full"
            >
              <Zap className="w-4 h-4 text-cyan-400 mr-2" />
              <span className="text-sm font-medium text-white">
                Next-Generation AI Vision Platform
              </span>
            </motion.div>

            {/* Main Heading */}
            <motion.h1
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1, duration: 0.8 }}
              className="text-5xl md:text-7xl lg:text-8xl font-display font-black mb-6 leading-tight"
            >
              <span className="block text-white">SIGHTLINE</span>
              <motion.span 
                key={currentText}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.5 }}
                className="block bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 bg-clip-text text-transparent"
              >
                {typedTexts[currentText]}
              </motion.span>
            </motion.h1>

            {/* Subtitle */}
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.2 }}
              className="text-xl md:text-2xl text-gray-300 mb-12 max-w-4xl mx-auto leading-relaxed"
            >
              Revolutionary AI-powered platform delivering{' '}
              <span className="text-cyan-400 font-semibold">real-time collision detection</span>,{' '}
              <span className="text-blue-400 font-semibold">driver monitoring</span>, and{' '}
              <span className="text-purple-400 font-semibold">autonomous safety systems</span>{' '}
              for the future of transportation.
            </motion.p>

            {/* CTA Button */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.4 }}
              className="mb-16"
            >
              <motion.a
                href="#contact"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="inline-flex items-center px-12 py-5 bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold rounded-xl hover:from-cyan-600 hover:to-blue-700 transition-all duration-300 shadow-lg hover:shadow-cyan-500/25 text-lg group"
              >
                <span>Contact Us</span>
                <motion.div
                  className="ml-2 inline-block"
                  animate={{ x: [0, 5, 0] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                >
                  →
                </motion.div>
              </motion.a>
            </motion.div>
          </div>
        </div>
        
        {/* Bottom Section - Technology Preview */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.8 }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl md:text-4xl font-display font-bold text-white mb-4">
            See Our Technology in Action
          </h2>
          <p className="text-lg text-gray-300 max-w-3xl mx-auto mb-8">
            Watch how our AI-powered systems work in real-world scenarios, providing instant threat detection and autonomous safety responses.
          </p>
          
          {/* Stats */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
            {stats.map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 2 + index * 0.1 }}
                className="text-center bg-gradient-to-br from-gray-800/50 to-gray-900/50 rounded-2xl p-6 border border-gray-700/50 backdrop-blur-sm"
              >
                <div className="text-3xl md:text-4xl font-display font-black bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent mb-2">
                  <CountUp
                    end={stat.number}
                    duration={2}
                    delay={2.2 + index * 0.1}
                    suffix={stat.suffix}
                  />
                </div>
                <div className="text-sm text-gray-400 font-medium uppercase tracking-wider">
                  {stat.label}
                </div>
              </motion.div>
            ))}
          </div>

          {/* Feature Icons */}
          <div className="flex flex-wrap items-center justify-center gap-6">
            {features.map((feature, index) => (
              <motion.div
                key={feature.label}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 2.4 + index * 0.1 }}
                whileHover={{ scale: 1.1, y: -5 }}
                className="flex items-center space-x-3 px-6 py-3 bg-gradient-to-br from-gray-800/50 to-gray-900/50 backdrop-blur-sm border border-gray-700/50 rounded-xl hover:border-cyan-500/50 transition-all duration-300"
              >
                <feature.icon className="w-5 h-5 text-cyan-400" />
                <span className="text-sm font-medium text-white">{feature.label}</span>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Scroll Indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 2.5 }}
        className="absolute bottom-8 left-1/2 transform -translate-x-1/2 z-20"
      >
        <motion.div
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
          className="flex flex-col items-center text-white/60"
        >
          <span className="text-xs font-medium mb-2 uppercase tracking-wider">Scroll to explore</span>
          <div className="w-6 h-10 border-2 border-cyan-500/30 rounded-full flex justify-center">
            <motion.div
              animate={{ y: [0, 12, 0] }}
              transition={{ duration: 2, repeat: Infinity }}
              className="w-1 h-3 bg-cyan-400/60 rounded-full mt-2"
            />
          </div>
        </motion.div>
      </motion.div>
    </section>
  )
}