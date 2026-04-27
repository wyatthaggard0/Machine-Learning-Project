import "./globals.css"

export const metadata = {
  title: "FRAUD.MODEL · TERMINAL",
  description: "IEEE-CIS Credit Card Fraud Detection · Real-time scoring terminal",
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
