'use client'

import { motion } from 'framer-motion'
import { useInView } from 'react-intersection-observer'
import { Check, Star, Zap } from 'lucide-react'

const plans = [
  {
    name: 'Professional',
    subtitle: 'For growing fleets',
    price: 299,
    period: '/vehicle/month',
    features: [
      'Up to 50 vehicles',
      'Real-time monitoring',
      'Basic analytics dashboard',
      'Email support',
      'Mobile app access',
      'Standard integrations'
    ],
    color: 'from-blue-500 to-cyan-500'
  },
  {
    name: 'Enterprise',
    subtitle: 'For large organizations',
    price: 599,
    period: '/vehicle/month',
    features: [
      'Unlimited vehicles',
      'Advanced AI features',
      'Custom analytics & reporting',
      '24/7 phone support',
      'API access & integrations',
      'Dedicated account manager',
      'Custom training',
      'Priority support'
    ],
    color: 'from-green-500 to-emerald-500',
    featured: true
  },
  {
    name: 'Government',
    subtitle: 'For public sector',
    price: null,
    period: 'Custom Pricing',
    features: [
      'Government compliance',
      'On-premise deployment',
      'Custom integrations',
      'White-label options',
      'Training & certification',
      'Security clearance',
      'Audit support'
    ],
    color: 'from-purple-500 to-pink-500'
  }
]

export default function PricingSection() {
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1
  })

  return (
    <section id="pricing" className="section-padding relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-br from-dark-950 via-dark-900 to-dark-950" />
      <div className="absolute inset-0 cyber-grid opacity-5" />

      <div className="container-custom relative z-10">
        <motion.div
          ref={ref}
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          className="text-center mb-20"
        >
          <div className="inline-flex items-center px-6 py-3 mb-6 bg-green-500/10 border border-green-500/20 rounded-full">
            <Star className="w-4 h-4 text-green-400 mr-2" />
            <span className="text-sm font-medium text-green-400">Transparent Pricing</span>
          </div>

          <h2 className="text-4xl md:text-6xl font-display font-black text-white mb-6">
            Enterprise-Grade
            <span className="block text-gradient">Plans</span>
          </h2>
          
          <p className="text-xl text-gray-400 max-w-3xl mx-auto">
            Flexible pricing designed for organizations of all sizes. All plans include 24/7 support and enterprise security.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-16">
          {plans.map((plan, index) => (
            <motion.div
              key={plan.name}
              initial={{ opacity: 0, y: 50 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: index * 0.2 }}
              className={`relative group ${plan.featured ? 'lg:scale-105' : ''}`}
            >
              {plan.featured && (
                <div className="absolute -top-4 left-1/2 transform -translate-x-1/2 z-10">
                  <div className="bg-gradient-to-r from-green-500 to-emerald-500 text-white px-6 py-2 rounded-full text-sm font-semibold">
                    Most Popular
                  </div>
                </div>
              )}

              <div className={`absolute -inset-0.5 bg-gradient-to-r ${plan.color} rounded-3xl blur opacity-0 group-hover:opacity-20 transition duration-1000`} />
              
              <div className={`relative card-glass h-full ${plan.featured ? 'border-green-500/30' : ''}`}>
                <div className="text-center mb-8">
                  <h3 className="text-2xl font-display font-bold text-white mb-2">
                    {plan.name}
                  </h3>
                  <p className="text-gray-400 mb-6">{plan.subtitle}</p>
                  
                  <div className="mb-6">
                    {plan.price ? (
                      <>
                        <span className="text-5xl font-display font-black text-gradient">
                          ${plan.price}
                        </span>
                        <span className="text-gray-400 text-lg">{plan.period}</span>
                      </>
                    ) : (
                      <span className="text-3xl font-display font-black text-gradient">
                        {plan.period}
                      </span>
                    )}
                  </div>
                </div>

                <ul className="space-y-4 mb-8">
                  {plan.features.map((feature, idx) => (
                    <li key={idx} className="flex items-center text-gray-300">
                      <Check className="w-5 h-5 text-green-400 mr-3 flex-shrink-0" />
                      {feature}
                    </li>
                  ))}
                </ul>

                <motion.a
                  href="#contact"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className={`block text-center w-full ${plan.featured ? 'btn-primary' : 'btn-secondary'}`}
                >
                  {plan.price ? 'Get Started' : 'Contact Sales'}
                </motion.a>
              </div>
            </motion.div>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ delay: 0.8 }}
          className="text-center"
        >
          <div className="inline-flex items-center px-6 py-3 bg-green-500/10 border border-green-500/20 rounded-full">
            <Check className="w-4 h-4 text-green-400 mr-2" />
            <span className="text-sm text-gray-300">
              30-day money-back guarantee • No setup fees • Cancel anytime
            </span>
          </div>
        </motion.div>
      </div>
    </section>
  )
}