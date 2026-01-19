'use client'

import { motion } from 'framer-motion'
import { useInView } from 'react-intersection-observer'
import { useState } from 'react'
import { Mail, Phone, MapPin, Send, Clock, Shield, Award } from 'lucide-react'

export default function ContactSection() {
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1
  })

  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    email: '',
    company: '',
    phone: '',
    fleetSize: '',
    solution: '',
    message: ''
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    // Handle form submission
    console.log('Form submitted:', formData)
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    })
  }

  return (
    <section id="contact" className="section-padding relative overflow-hidden bg-gradient-to-br from-gray-900 via-black to-gray-900">
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
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-start">
          {/* Contact Info */}
          <motion.div
            ref={ref}
            initial={{ opacity: 0, x: -50 }}
            animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.8 }}
          >
            <div className="mb-12">
              <h2 className="text-4xl md:text-5xl font-display font-black text-white mb-6">
                Ready to Transform Your
                <span className="block bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 bg-clip-text text-transparent">Fleet Safety?</span>
              </h2>
              <p className="text-xl text-gray-300 leading-relaxed">
                Join the next generation of intelligent transportation with SIGHTLINE's 
                cutting-edge AI vision platform. Experience the future of vehicle safety today.
              </p>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-3 gap-6 mb-12">
              <div className="text-center p-6 bg-gradient-to-br from-gray-800/50 to-gray-900/50 backdrop-blur-sm border border-gray-700/50 rounded-2xl">
                <Clock className="w-8 h-8 text-cyan-400 mx-auto mb-4" />
                <div className="text-2xl font-display font-black bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent mb-2">24/7</div>
                <div className="text-sm text-gray-400">Enterprise Support</div>
              </div>
              <div className="text-center p-6 bg-gradient-to-br from-gray-800/50 to-gray-900/50 backdrop-blur-sm border border-gray-700/50 rounded-2xl">
                <Shield className="w-8 h-8 text-green-400 mx-auto mb-4" />
                <div className="text-2xl font-display font-black bg-gradient-to-r from-green-400 to-emerald-500 bg-clip-text text-transparent mb-2">30 Days</div>
                <div className="text-sm text-gray-400">Money-Back Guarantee</div>
              </div>
              <div className="text-center p-6 bg-gradient-to-br from-gray-800/50 to-gray-900/50 backdrop-blur-sm border border-gray-700/50 rounded-2xl">
                <Award className="w-8 h-8 text-purple-400 mx-auto mb-4" />
                <div className="text-2xl font-display font-black bg-gradient-to-r from-purple-400 to-pink-500 bg-clip-text text-transparent mb-2">99.9%</div>
                <div className="text-sm text-gray-400">Uptime SLA</div>
              </div>
            </div>

            {/* Contact Methods */}
            <div className="space-y-6">
              <div className="flex items-center space-x-4">
                <div className="w-12 h-12 bg-gradient-to-r from-primary-500 to-cyber-500 rounded-xl flex items-center justify-center">
                  <Phone className="w-6 h-6 text-white" />
                </div>
                <div>
                  <div className="text-white font-semibold">Sales Hotline</div>
                  <div className="text-gray-400">+1 (555) 123-4567</div>
                </div>
              </div>
              
              <div className="flex items-center space-x-4">
                <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-emerald-500 rounded-xl flex items-center justify-center">
                  <Mail className="w-6 h-6 text-white" />
                </div>
                <div>
                  <div className="text-white font-semibold">Enterprise Sales</div>
                  <div className="text-gray-400">sales@smartvehicle.ai</div>
                </div>
              </div>
              
              <div className="flex items-center space-x-4">
                <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl flex items-center justify-center">
                  <MapPin className="w-6 h-6 text-white" />
                </div>
                <div>
                  <div className="text-white font-semibold">Headquarters</div>
                  <div className="text-gray-400">Silicon Valley, CA</div>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Contact Form */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="bg-gradient-to-br from-gray-800/50 to-gray-900/50 backdrop-blur-sm border border-gray-700/50 rounded-2xl p-8"
          >
            <div className="text-center mb-8">
              <h3 className="text-2xl font-display font-bold text-white mb-2">
                Get Started Today
              </h3>
              <p className="text-gray-400">
                Contact us to learn more about SIGHTLINE
              </p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    First Name *
                  </label>
                  <input
                    type="text"
                    name="firstName"
                    required
                    value={formData.firstName}
                    onChange={handleChange}
                    className="w-full px-4 py-3 bg-gray-800/50 border border-gray-700/50 rounded-xl text-white placeholder-gray-400 focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/20 transition-all duration-300"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Last Name *
                  </label>
                  <input
                    type="text"
                    name="lastName"
                    required
                    value={formData.lastName}
                    onChange={handleChange}
                    className="w-full px-4 py-3 bg-gray-800/50 border border-gray-700/50 rounded-xl text-white placeholder-gray-400 focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/20 transition-all duration-300"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Business Email *
                </label>
                <input
                  type="email"
                  name="email"
                  required
                  value={formData.email}
                  onChange={handleChange}
                  className="w-full px-4 py-3 bg-gray-800/50 border border-gray-700/50 rounded-xl text-white placeholder-gray-400 focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/20 transition-all duration-300"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Company Name *
                </label>
                <input
                  type="text"
                  name="company"
                  required
                  value={formData.company}
                  onChange={handleChange}
                  className="w-full px-4 py-3 bg-gray-800/50 border border-gray-700/50 rounded-xl text-white placeholder-gray-400 focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/20 transition-all duration-300"
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Phone Number
                  </label>
                  <input
                    type="tel"
                    name="phone"
                    value={formData.phone}
                    onChange={handleChange}
                    className="w-full px-4 py-3 bg-gray-800/50 border border-gray-700/50 rounded-xl text-white placeholder-gray-400 focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/20 transition-all duration-300"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Fleet Size
                  </label>
                  <select
                    name="fleetSize"
                    value={formData.fleetSize}
                    onChange={handleChange}
                    className="w-full px-4 py-3 bg-gray-800/50 border border-gray-700/50 rounded-xl text-white focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/20 transition-all duration-300 [&>option]:bg-gray-800 [&>option]:text-white"
                  >
                    <option value="" className="bg-gray-800 text-white">Select size</option>
                    <option value="1-10" className="bg-gray-800 text-white">1-10 vehicles</option>
                    <option value="11-50" className="bg-gray-800 text-white">11-50 vehicles</option>
                    <option value="51-200" className="bg-gray-800 text-white">51-200 vehicles</option>
                    <option value="200+" className="bg-gray-800 text-white">200+ vehicles</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Primary Interest *
                </label>
                <select
                  name="solution"
                  required
                  value={formData.solution}
                  onChange={handleChange}
                  className="w-full px-4 py-3 bg-gray-800/50 border border-gray-700/50 rounded-xl text-white focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/20 transition-all duration-300 [&>option]:bg-gray-800 [&>option]:text-white"
                >
                  <option value="" className="bg-gray-800 text-white">Select solution</option>
                  <option value="fleet" className="bg-gray-800 text-white">Fleet Management</option>
                  <option value="enforcement" className="bg-gray-800 text-white">Law Enforcement</option>
                  <option value="smart-city" className="bg-gray-800 text-white">Smart Cities</option>
                  <option value="custom" className="bg-gray-800 text-white">Custom Solution</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Additional Requirements
                </label>
                <textarea
                  name="message"
                  rows={4}
                  value={formData.message}
                  onChange={handleChange}
                  placeholder="Tell us about your specific needs..."
                  className="w-full px-4 py-3 bg-gray-800/50 border border-gray-700/50 rounded-xl text-white placeholder-gray-400 focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/20 transition-all duration-300 resize-none"
                />
              </div>

              <motion.button
                type="submit"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="w-full inline-flex items-center justify-center px-8 py-4 bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold rounded-xl hover:from-cyan-600 hover:to-blue-700 transition-all duration-300 shadow-lg hover:shadow-cyan-500/25 text-lg"
              >
                <Send className="w-5 h-5 mr-2" />
                Contact Us
              </motion.button>

              <p className="text-xs text-gray-500 text-center">
                By submitting this form, you agree to our privacy policy and terms of service.
              </p>
            </form>
          </motion.div>
        </div>
      </div>
    </section>
  )
}