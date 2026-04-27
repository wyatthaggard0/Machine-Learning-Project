import "./globals.css"

export const metadata = {
  title: "Fraud Detection Dashboard",
  description: "IEEE-CIS Credit Card Fraud Detection",
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
