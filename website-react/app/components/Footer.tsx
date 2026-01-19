'use client'

import { motion } from 'framer-motion'
import { Zap, Mail, Phone, MapPin } from 'lucide-react'

const footerLinks = {
  Platform: [
    { name: 'Features', href: '#platform' },
    { name: 'Technology', href: '#technology' },
    { name: 'Pricing', href: '#pricing' },
    { name: 'API Documentation', href: '#' }
  ],
  Company: [
    { name: 'About Us', href: '#' },
    { name: 'Careers', href: '#' },
    { name: 'Press', href: '#' },
    { name: 'Partners', href: '#' }
  ],
  Support: [
    { name: 'Help Center', href: '#' },
    { name: 'Contact Support', href: '#' },
    { name: 'System Status', href: '#' },
    { name: 'Security', href: '#' }
  ]
}

export default function Footer() {
  return (
    <footer className="relative bg-dark-950 border-t border-white/10">
      <div className="absolute inset-0 cyber-grid opacity-5" />
      
      <div className="container-custom relative z-10 py-16">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-12 mb-12">
          {/* Brand */}
          <div className="lg:col-span-1">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="mb-6"
            >
              <div className="flex items-center space-x-3 mb-4">
                <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-cyber-500 rounded-xl flex items-center justify-center">
                  <Zap className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h3 className="text-xl font-display font-bold text-white">SmartVehicle</h3>
                  <p className="text-xs text-gray-400 -mt-1">Intelligence</p>
                </div>
              </div>
              <p className="text-gray-400 leading-relaxed max-w-sm">
                Enterprise AI vehicle safety platform trusted by Fortune 500 companies worldwide.
              </p>
            </motion.div>

            {/* Contact Info */}
            <div className="space-y-3">
              <div className="flex items-center space-x-3 text-sm text-gray-400">
                <Mail className="w-4 h-4" />
                <span>sales@smartvehicle.ai</span>
              </div>
              <div className="flex items-center space-x-3 text-sm text-gray-400">
                <Phone className="w-4 h-4" />
                <span>+1 (555) 123-4567</span>
              </div>
              <div className="flex items-center space-x-3 text-sm text-gray-400">
                <MapPin className="w-4 h-4" />
                <span>Silicon Valley, CA</span>
              </div>
            </div>
          </div>

          {/* Links */}
          {Object.entries(footerLinks).map(([category, links], index) => (
            <motion.div
              key={category}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
            >
              <h4 className="text-white font-semibold mb-6">{category}</h4>
              <ul className="space-y-3">
                {links.map((link) => (
                  <li key={link.name}>
                    <a
                      href={link.href}
                      className="text-gray-400 hover:text-primary-400 transition-colors duration-300"
                    >
                      {link.name}
                    </a>
                  </li>
                ))}
              </ul>
            </motion.div>
          ))}
        </div>

        {/* Bottom */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="pt-8 border-t border-white/10 flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0"
        >
          <p className="text-gray-500 text-sm">
            © 2024 SmartVehicle Intelligence. All rights reserved.
          </p>
          <div className="flex space-x-6 text-sm">
            <a href="#" className="text-gray-500 hover:text-primary-400 transition-colors">
              Privacy Policy
            </a>
            <a href="#" className="text-gray-500 hover:text-primary-400 transition-colors">
              Terms of Service
            </a>
            <a href="#" className="text-gray-500 hover:text-primary-400 transition-colors">
              Security
            </a>
          </div>
        </motion.div>
      </div>
    </footer>
  )
}