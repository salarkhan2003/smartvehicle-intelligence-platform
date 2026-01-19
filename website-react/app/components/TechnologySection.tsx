'use client'

import { motion } from 'framer-motion'
import { useInView } from 'react-intersection-observer'
import { Brain, Eye, Cpu, Database, Cloud, Zap } from 'lucide-react'

const technologies = [
  {
    category: 'AI & Machine Learning',
    icon: Brain,
    color: 'from-purple-500 to-pink-500',
    items: [
      { name: 'YOLOv8', description: 'Real-time object detection' },
      { name: 'MediaPipe', description: 'Face & pose estimation' },
      { name: 'TensorFlow', description: 'Deep learning models' },
      { name: 'OpenCV', description: 'Computer vision' }
    ]
  },
  {
    category: 'Edge Computing',
    icon: Cpu,
    color: 'from-blue-500 to-cyan-500',
    items: [
      { name: 'NVIDIA Jetson', description: 'GPU acceleration' },
      { name: 'Intel OpenVINO', description: 'Model optimization' },
      { name: 'ARM Cortex', description: 'Low-power processing' },
      { name: 'FPGA', description: 'Custom acceleration' }
    ]
  },
  {
    category: 'Cloud Infrastructure',
    icon: Cloud,
    color: 'from-green-500 to-emerald-500',
    items: [
      { name: 'AWS IoT', description: 'Device connectivity' },
      { name: 'Azure ML', description: 'Model training' },
      { name: 'Google Cloud', description: 'Data analytics' },
      { name: 'Kubernetes', description: 'Container orchestration' }
    ]
  }
]

export default function TechnologySection() {
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1
  })

  return (
    <section id="technology" className="section-padding relative overflow-hidden bg-gradient-to-br from-gray-900 via-black to-gray-900">
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
        <motion.div
          ref={ref}
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          className="text-center mb-20"
        >
            <div className="inline-flex items-center px-6 py-3 mb-6 bg-cyan-500/10 border border-cyan-500/20 rounded-full">
            <Zap className="w-4 h-4 text-cyan-400 mr-2" />
            <span className="text-sm font-medium text-cyan-400">Technology Stack</span>
          </div>

          <h2 className="text-4xl md:text-6xl font-display font-black text-white mb-6">
            See Our Technology
            <span className="block bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 bg-clip-text text-transparent">in Action</span>
          </h2>
          
          <p className="text-xl text-gray-300 max-w-3xl mx-auto mb-8">
            Watch how our AI-powered systems work in real-world scenarios, providing instant threat detection and autonomous safety responses.
          </p>
          
          {/* Video Demo */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={inView ? { opacity: 1, scale: 1 } : {}}
            transition={{ delay: 0.5, duration: 0.8 }}
            className="max-w-4xl mx-auto mb-12"
          >
            <div className="relative rounded-2xl overflow-hidden border border-cyan-500/30 shadow-2xl">
              <video
                autoPlay
                muted
                loop
                playsInline
                className="w-full h-auto"
                poster="/assets/video-poster.jpg"
              >
                <source src="/assets/304445.mp4" type="video/mp4" />
                Your browser does not support the video tag.
              </video>
              
              {/* Video Overlay */}
              <div className="absolute inset-0 bg-gradient-to-t from-black/20 via-transparent to-transparent pointer-events-none" />
              
              {/* Play Button Overlay */}
              <div className="absolute inset-0 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity duration-300 bg-black/20">
                <div className="w-16 h-16 bg-cyan-500/80 rounded-full flex items-center justify-center backdrop-blur-sm">
                  <div className="w-0 h-0 border-l-[12px] border-l-white border-y-[8px] border-y-transparent ml-1"></div>
                </div>
              </div>
            </div>
          </motion.div>
          
          <p className="text-lg text-gray-400 max-w-3xl mx-auto">
            Built with industry-leading AI and computer vision technologies for maximum performance and reliability.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {technologies.map((tech, index) => (
            <motion.div
              key={tech.category}
              initial={{ opacity: 0, y: 50 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: index * 0.2 }}
              className="bg-gradient-to-br from-gray-800/50 to-gray-900/50 backdrop-blur-sm border border-gray-700/50 rounded-2xl p-8 group hover:scale-105 hover:border-cyan-500/50 transition-all duration-300"
            >
              <div className={`w-16 h-16 bg-gradient-to-r ${tech.color} rounded-xl flex items-center justify-center mb-6`}>
                <tech.icon className="w-8 h-8 text-white" />
              </div>
              
              <h3 className="text-2xl font-display font-bold text-white mb-6">
                {tech.category}
              </h3>
              
              <div className="space-y-4">
                {tech.items.map((item, idx) => (
                  <div key={idx} className="flex items-start space-x-3">
                    <div className={`w-2 h-2 bg-gradient-to-r ${tech.color} rounded-full mt-2 flex-shrink-0`} />
                    <div>
                      <div className="text-white font-semibold">{item.name}</div>
                      <div className="text-gray-400 text-sm">{item.description}</div>
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}