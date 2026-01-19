'use client'

import { useEffect, useState } from 'react'
import { motion, useScroll, useTransform } from 'framer-motion'
import { useInView } from 'react-intersection-observer'

// Components
import Navbar from './components/Navbar'
import HeroSection from './components/HeroSection'
import FeaturesSection from './components/FeaturesSection'
import TechnologySection from './components/TechnologySection'
import SolutionsSection from './components/SolutionsSection'
import ContactSection from './components/ContactSection'
import Footer from './components/Footer'
import LoadingScreen from './components/LoadingScreen'

export default function Home() {
  const [isLoading, setIsLoading] = useState(true)
  const { scrollYProgress } = useScroll()
  
  // Parallax effects
  const backgroundY = useTransform(scrollYProgress, [0, 1], ['0%', '100%'])
  const textY = useTransform(scrollYProgress, [0, 1], ['0%', '200%'])

  useEffect(() => {
    // Simulate loading time
    const timer = setTimeout(() => {
      setIsLoading(false)
    }, 3000)

    return () => clearTimeout(timer)
  }, [])

  if (isLoading) {
    return <LoadingScreen />
  }

  return (
    <main className="relative min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 overflow-x-hidden">
      {/* Background Grid */}
      <div className="fixed inset-0 opacity-10 pointer-events-none">
        <div className="absolute inset-0" style={{
          backgroundImage: `
            linear-gradient(rgba(6, 182, 212, 0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(6, 182, 212, 0.1) 1px, transparent 1px)
          `,
          backgroundSize: '50px 50px'
        }} />
      </div>
      
      {/* Gradient Overlays */}
      <div className="fixed inset-0 bg-gradient-to-br from-cyan-950/20 via-transparent to-blue-950/20 pointer-events-none" />
      
      {/* Navigation */}
      <Navbar />
      
      {/* Hero Section */}
      <HeroSection />
      
      {/* Features Section */}
      <FeaturesSection />
      
      {/* Technology Section */}
      <TechnologySection />
      
      {/* Solutions Section */}
      <SolutionsSection />
      
      {/* Contact Section */}
      <ContactSection />
      
      {/* Footer */}
      <Footer />
      
      {/* Scroll Progress Indicator */}
      <motion.div
        className="fixed top-0 left-0 right-0 h-1 bg-gradient-to-r from-cyan-500 to-blue-500 transform-gpu z-50"
        style={{ scaleX: scrollYProgress, transformOrigin: '0%' }}
      />
    </main>
  )
}