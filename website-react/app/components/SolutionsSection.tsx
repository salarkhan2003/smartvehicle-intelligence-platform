'use client'

import { motion } from 'framer-motion'
import { useInView } from 'react-intersection-observer'
import { useRef, useState } from 'react'
import { 
  Truck, 
  Shield, 
  Building, 
  Users, 
  TrendingUp, 
  Clock, 
  Award,
  Play,
  Pause
} from 'lucide-react'

const solutions = [
  {
    id: 1,
    icon: Truck,
    title: 'Fleet Management',
    subtitle: 'Commercial & Logistics',
    description: 'Comprehensive driver monitoring and vehicle safety for commercial fleets. Reduce accidents by 67% and insurance costs by 45%.',
    features: [
      'Real-time driver fatigue detection',
      'Automated incident reporting', 
      'Fleet-wide analytics dashboard',
      'Insurance integration & discounts'
    ],
    results: [
      { metric: '67%', label: 'Accident Reduction' },
      { metric: '45%', label: 'Insurance Savings' }
    ],
    color: 'from-blue-500 to-cyan-500'
  },
  {
    id: 2,
    icon: Shield,
    title: 'Law Enforcement',
    subtitle: 'Traffic & Public Safety',
    description: 'Advanced ANPR and automated enforcement for traffic management. Increase citation accuracy by 89% and reduce processing time by 78%.',
    features: [
      'License plate recognition (OCR)',
      'Speed enforcement automation',
      'Evidence capture & storage', 
      'Court-ready documentation'
    ],
    results: [
      { metric: '89%', label: 'Citation Accuracy' },
      { metric: '78%', label: 'Time Savings' }
    ],
    color: 'from-green-500 to-emerald-500',
    featured: true
  },
  {
    id: 3,
    icon: Building,
    title: 'Smart Cities',
    subtitle: 'Urban Infrastructure',
    description: 'Integrated traffic management and safety monitoring for urban environments. Improve traffic flow by 34% and reduce response times by 56%.',
    features: [
      'Traffic flow optimization',
      'Accident prevention systems',
      'Emergency response integration',
      'City-wide analytics platform'
    ],
    results: [
      { metric: '34%', label: 'Traffic Improvement' },
      { metric: '56%', label: 'Faster Response' }
    ],
    color: 'from-purple-500 to-pink-500'
  }
]

export default function SolutionsSection() {
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1
  })
  
  const videoRef = useRef<HTMLVideoElement>(null)
  const [isPlaying, setIsPlaying] = useState(true)

  const toggleVideo = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause()
      } else {
        videoRef.current.play()
      }
      setIsPlaying(!isPlaying)
    }
  }

  return (
    <section id="solutions" className="section-padding relative overflow-hidden bg-gradient-to-br from-gray-900 via-black to-gray-900">
      {/* Background Elements */}
      <div className="absolute inset-0 opacity-10">
        <div className="absolute inset-0" style={{
          backgroundImage: `
            linear-gradient(rgba(6, 182, 212, 0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(6, 182, 212, 0.1) 1px, transparent 1px)
          `,
          backgroundSize: '50px 50px'
        }} />
      </div>

      <div className="container-custom relative z-10">
        {/* Section Header */}
        <motion.div
          ref={ref}
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
          className="text-center mb-20"
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={inView ? { opacity: 1, scale: 1 } : {}}
            transition={{ delay: 0.2 }}
            className="inline-flex items-center px-6 py-3 mb-6 bg-cyan-500/10 border border-cyan-500/20 rounded-full"
          >
            <Building className="w-4 h-4 text-cyan-400 mr-2" />
            <span className="text-sm font-medium text-cyan-400">Industry Solutions</span>
          </motion.div>

          <h2 className="text-4xl md:text-6xl font-display font-black text-white mb-6">
            Built for Enterprise
            <span className="block bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 bg-clip-text text-transparent">Scale</span>
          </h2>
          
          <p className="text-xl text-gray-400 max-w-3xl mx-auto leading-relaxed">
            Tailored solutions for different industries with proven ROI, 
            enterprise-grade support, and seamless integration capabilities.
          </p>
        </motion.div>

        {/* Video Section */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ delay: 0.4 }}
          className="relative mb-20 rounded-3xl overflow-hidden"
        >
          <div className="relative aspect-video">
            <video
              ref={videoRef}
              autoPlay
              muted
              loop
              playsInline
              className="w-full h-full object-cover"
            >
              <source src="/assets/304445.mp4" type="video/mp4" />
              Your browser does not support the video tag.
            </video>
            
            {/* Video Overlay */}
            <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent" />
            
            {/* Video Controls */}
            <motion.button
              onClick={toggleVideo}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              className="absolute top-6 right-6 w-12 h-12 bg-white/20 backdrop-blur-sm border border-white/30 rounded-full flex items-center justify-center text-white hover:bg-white/30 transition-all duration-300"
            >
              {isPlaying ? <Pause size={20} /> : <Play size={20} />}
            </motion.button>

            {/* Video Content Overlay */}
            <div className="absolute bottom-8 left-8 right-8">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={inView ? { opacity: 1, y: 0 } : {}}
                transition={{ delay: 0.8 }}
                className="max-w-2xl"
              >
                <h3 className="text-2xl md:text-3xl font-display font-bold text-white mb-4">
                  Real-World AI Implementation
                </h3>
                <p className="text-gray-300 text-lg leading-relaxed">
                  Experience SIGHTLINE's advanced AI systems in action, delivering 
                  instant threat detection and autonomous safety responses across various scenarios.
                </p>
              </motion.div>
            </div>
          </div>
        </motion.div>

        {/* Solutions Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-20">
          {solutions.map((solution, index) => (
            <motion.div
              key={solution.id}
              initial={{ opacity: 0, y: 50 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: 0.6 + index * 0.2 }}
              className={`group relative ${solution.featured ? 'lg:scale-105' : ''}`}
            >
              {/* Featured Badge */}
              {solution.featured && (
                <div className="absolute -top-4 left-1/2 transform -translate-x-1/2 z-10">
                  <div className="bg-gradient-to-r from-green-500 to-emerald-500 text-white px-6 py-2 rounded-full text-sm font-semibold">
                    Most Popular
                  </div>
                </div>
              )}

              {/* Glow Effect */}
              <div className={`absolute -inset-0.5 bg-gradient-to-r ${solution.color} rounded-3xl blur opacity-0 group-hover:opacity-20 transition duration-1000`} />
              
              {/* Card */}
              <div className={`relative bg-gradient-to-br from-gray-800/50 to-gray-900/50 backdrop-blur-sm border border-gray-700/50 rounded-2xl p-8 h-full hover:border-cyan-500/50 transition-all duration-300 ${solution.featured ? 'border-cyan-500/30' : ''}`}>
                {/* Header */}
                <div className="text-center mb-8">
                  <div className={`w-20 h-20 bg-gradient-to-r ${solution.color} rounded-2xl flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform duration-300`}>
                    <solution.icon className="w-10 h-10 text-white" />
                  </div>
                  
                  <h3 className="text-2xl font-display font-bold text-white mb-2">
                    {solution.title}
                  </h3>
                  
                  <p className="text-sm text-gray-400 uppercase tracking-wider font-medium">
                    {solution.subtitle}
                  </p>
                </div>

                {/* Description */}
                <p className="text-gray-400 mb-8 leading-relaxed">
                  {solution.description}
                </p>

                {/* Features */}
                <ul className="space-y-4 mb-8">
                  {solution.features.map((feature, idx) => (
                    <li key={idx} className="flex items-start text-sm text-gray-300">
                      <div className={`w-2 h-2 bg-gradient-to-r ${solution.color} rounded-full mr-3 mt-2 flex-shrink-0`} />
                      {feature}
                    </li>
                  ))}
                </ul>

                {/* Results */}
                <div className="grid grid-cols-2 gap-4 mb-8 p-6 bg-gray-800/30 rounded-xl border border-gray-700/30">
                  {solution.results.map((result, idx) => (
                    <div key={idx} className="text-center">
                      <div className={`text-2xl font-display font-black bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent mb-1`}>
                        {result.metric}
                      </div>
                      <div className="text-xs text-gray-400 uppercase tracking-wider">
                        {result.label}
                      </div>
                    </div>
                  ))}
                </div>

                {/* CTA */}
                <motion.a
                  href="#contact"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className={`block text-center ${solution.featured ? 'inline-flex items-center justify-center px-8 py-4 bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold rounded-xl hover:from-cyan-600 hover:to-blue-700 transition-all duration-300 shadow-lg hover:shadow-cyan-500/25' : 'inline-flex items-center justify-center px-8 py-4 bg-transparent border-2 border-cyan-500 text-cyan-400 font-semibold rounded-xl hover:bg-cyan-500 hover:text-white transition-all duration-300'} w-full`}
                >
                  Contact Us
                </motion.a>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Bottom Stats */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ delay: 1.4 }}
          className="text-center"
        >
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-12">
            <div className="text-center">
              <TrendingUp className="w-8 h-8 text-green-400 mx-auto mb-4" />
              <div className="text-2xl font-display font-black bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent mb-2">67%</div>
              <div className="text-sm text-gray-400">Accident Reduction</div>
            </div>
            <div className="text-center">
              <Clock className="w-8 h-8 text-cyan-400 mx-auto mb-4" />
              <div className="text-2xl font-display font-black bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent mb-2">24/7</div>
              <div className="text-sm text-gray-400">Support</div>
            </div>
            <div className="text-center">
              <Award className="w-8 h-8 text-yellow-400 mx-auto mb-4" />
              <div className="text-2xl font-display font-black bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent mb-2">99.9%</div>
              <div className="text-sm text-gray-400">Uptime SLA</div>
            </div>
          </div>

          <motion.a
            href="#contact"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="inline-flex items-center px-12 py-5 bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold rounded-xl hover:from-cyan-600 hover:to-blue-700 transition-all duration-300 shadow-lg hover:shadow-cyan-500/25 text-lg"
          >
            Get Started Today
          </motion.a>
        </motion.div>
      </div>
    </section>
  )
}