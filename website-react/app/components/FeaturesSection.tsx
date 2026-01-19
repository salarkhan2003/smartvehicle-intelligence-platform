'use client'

import { motion } from 'framer-motion'
import { useInView } from 'react-intersection-observer'
import { 
  Eye, 
  Shield, 
  Brain, 
  Zap, 
  Camera, 
  AlertTriangle,
  Bell,
  Cpu,
  Car,
  Users,
  Activity,
  Target
} from 'lucide-react'

const features = [
  {
    id: 1,
    icon: Eye,
    title: 'Advanced Computer Vision',
    description: 'YOLOv8-powered real-time detection of 80+ object classes including vehicles, pedestrians, and wildlife',
    features: ['Real-time object detection', 'Distance estimation', 'Threat assessment', 'Animal detection'],
    color: 'from-blue-500 to-cyan-500',
    delay: 0.1
  },
  {
    id: 2,
    icon: Brain,
    title: 'Driver Intelligence System',
    description: 'MediaPipe-based fatigue detection with advanced eye tracking and behavioral analysis',
    features: ['Drowsiness detection', 'Attention monitoring', 'Fatigue scoring', 'Head pose tracking'],
    color: 'from-purple-500 to-pink-500',
    delay: 0.2
  },
  {
    id: 3,
    icon: Shield,
    title: 'Automated Enforcement',
    description: 'OCR-powered license plate recognition with automated violation detection and documentation',
    features: ['License plate recognition', 'Speed enforcement', 'Violation logging', 'Evidence capture'],
    color: 'from-green-500 to-emerald-500',
    delay: 0.3
  },
  {
    id: 4,
    icon: AlertTriangle,
    title: 'Collision Prevention',
    description: 'Advanced trajectory prediction with real-time collision warnings and emergency braking',
    features: ['Collision warnings', 'Blind spot detection', 'Emergency braking', 'Path prediction'],
    color: 'from-red-500 to-orange-500',
    delay: 0.4
  },
  {
    id: 5,
    icon: Bell,
    title: 'Multi-Modal Alerts',
    description: 'Intelligent alert system with visual, audio, and haptic feedback based on severity levels',
    features: ['Visual alerts', 'Audio warnings', 'Haptic feedback', 'Voice announcements'],
    color: 'from-yellow-500 to-amber-500',
    delay: 0.5
  },
  {
    id: 6,
    icon: Cpu,
    title: 'Edge AI Processing',
    description: 'Real-time AI processing at the edge with cloud synchronization and continuous learning',
    features: ['Edge computing', 'Cloud sync', 'Continuous learning', 'Performance optimization'],
    color: 'from-indigo-500 to-blue-500',
    delay: 0.6
  }
]

const stats = [
  { number: '35+', label: 'AI Features', icon: Brain },
  { number: '99.7%', label: 'Accuracy', icon: Target },
  { number: '30ms', label: 'Response Time', icon: Zap },
  { number: '24/7', label: 'Monitoring', icon: Activity },
]

export default function FeaturesSection() {
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1
  })

  return (
    <section id="platform" className="section-padding relative overflow-hidden bg-gradient-to-br from-gray-900 via-black to-gray-900">
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
      <div className="absolute top-1/2 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl" />

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
            <Zap className="w-4 h-4 text-cyan-400 mr-2" />
            <span className="text-sm font-medium text-cyan-400">Platform Overview</span>
          </motion.div>

          <h2 className="text-4xl md:text-6xl font-display font-black text-white mb-6">
            Complete AI Safety
            <span className="block bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 bg-clip-text text-transparent">Ecosystem</span>
          </h2>
          
          <p className="text-xl text-gray-400 max-w-3xl mx-auto leading-relaxed">
            Six integrated modules delivering comprehensive vehicle intelligence with 
            enterprise-grade reliability, real-time processing, and autonomous decision-making capabilities.
          </p>
        </motion.div>

        {/* Stats Grid */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ delay: 0.4 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-20"
        >
          {stats.map((stat, index) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={inView ? { opacity: 1, scale: 1 } : {}}
              transition={{ delay: 0.6 + index * 0.1 }}
              className="text-center p-6 bg-gradient-to-br from-gray-800/50 to-gray-900/50 backdrop-blur-sm border border-gray-700/50 rounded-2xl hover:scale-105 hover:border-cyan-500/50 transition-all duration-300"
            >
              <stat.icon className="w-8 h-8 text-cyan-400 mx-auto mb-4" />
              <div className="text-3xl font-display font-black bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent mb-2">
                {stat.number}
              </div>
              <div className="text-sm text-gray-400 font-medium uppercase tracking-wider">
                {stat.label}
              </div>
            </motion.div>
          ))}
        </motion.div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <motion.div
              key={feature.id}
              initial={{ opacity: 0, y: 50 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: feature.delay, duration: 0.8 }}
              whileHover={{ y: -10, scale: 1.02 }}
              className="group relative"
            >
              {/* Glow Effect */}
              <div className={`absolute -inset-0.5 bg-gradient-to-r ${feature.color} rounded-2xl blur opacity-0 group-hover:opacity-20 transition duration-1000`} />
              
              {/* Card */}
              <div className="relative bg-gradient-to-br from-gray-800/50 to-gray-900/50 backdrop-blur-sm border border-gray-700/50 rounded-2xl p-8 h-full hover:border-cyan-500/50 transition-all duration-300">
                {/* Icon */}
                <div className={`w-16 h-16 bg-gradient-to-r ${feature.color} rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300`}>
                  <feature.icon className="w-8 h-8 text-white" />
                </div>

                {/* Content */}
                <h3 className="text-2xl font-display font-bold text-white mb-4 group-hover:bg-gradient-to-r group-hover:from-cyan-400 group-hover:to-blue-500 group-hover:bg-clip-text group-hover:text-transparent transition-all duration-300">
                  {feature.title}
                </h3>
                
                <p className="text-gray-400 mb-6 leading-relaxed">
                  {feature.description}
                </p>

                {/* Feature List */}
                <ul className="space-y-3">
                  {feature.features.map((item, idx) => (
                    <motion.li
                      key={idx}
                      initial={{ opacity: 0, x: -20 }}
                      animate={inView ? { opacity: 1, x: 0 } : {}}
                      transition={{ delay: feature.delay + 0.2 + idx * 0.1 }}
                      className="flex items-center text-sm text-gray-300"
                    >
                      <div className={`w-2 h-2 bg-gradient-to-r ${feature.color} rounded-full mr-3 flex-shrink-0`} />
                      {item}
                    </motion.li>
                  ))}
                </ul>

                {/* Hover Effect */}
                <motion.div
                  className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-transparent via-cyan-500 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"
                  layoutId={`feature-${feature.id}`}
                />
              </div>
            </motion.div>
          ))}
        </div>

        {/* Bottom CTA */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ delay: 1.2 }}
          className="text-center mt-20"
        >
          <motion.a
            href="#technology"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="inline-flex items-center px-8 py-4 bg-transparent border-2 border-cyan-500 text-cyan-400 font-semibold rounded-xl hover:bg-cyan-500 hover:text-white transition-all duration-300 transform hover:scale-105"
          >
            Explore Technology Stack
          </motion.a>
        </motion.div>
      </div>
    </section>
  )
}