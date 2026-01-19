import type { Metadata } from 'next'
import { Inter, Space_Grotesk } from 'next/font/google'
import './globals.css'

const inter = Inter({ 
  subsets: ['latin'],
  variable: '--font-inter',
})

const spaceGrotesk = Space_Grotesk({ 
  subsets: ['latin'],
  variable: '--font-space-grotesk',
})

export const metadata: Metadata = {
  title: 'SmartVehicle Intelligence | Autonomous Mobility Platform',
  description: 'Revolutionary AI-powered autonomous vehicle intelligence platform. Advanced collision detection, driver monitoring, and smart mobility solutions for the future of transportation.',
  keywords: 'autonomous vehicles, AI, smart mobility, collision detection, driver monitoring, autonomous driving, smart cars, vehicle intelligence',
  authors: [{ name: 'SmartVehicle Intelligence' }],
  creator: 'SmartVehicle Intelligence',
  publisher: 'SmartVehicle Intelligence',
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  metadataBase: new URL('https://smartvehicle-intelligence.vercel.app'),
  alternates: {
    canonical: '/',
  },
  openGraph: {
    title: 'SmartVehicle Intelligence | Autonomous Mobility Platform',
    description: 'Revolutionary AI-powered autonomous vehicle intelligence platform for the future of transportation.',
    url: 'https://smartvehicle-intelligence.vercel.app',
    siteName: 'SmartVehicle Intelligence',
    images: [
      {
        url: '/og-image.jpg',
        width: 1200,
        height: 630,
        alt: 'SmartVehicle Intelligence Platform',
      },
    ],
    locale: 'en_US',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'SmartVehicle Intelligence | Autonomous Mobility Platform',
    description: 'Revolutionary AI-powered autonomous vehicle intelligence platform for the future of transportation.',
    images: ['/og-image.jpg'],
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={`${inter.variable} ${spaceGrotesk.variable}`}>
      <body className="font-sans">
        {children}
      </body>
    </html>
  )
}