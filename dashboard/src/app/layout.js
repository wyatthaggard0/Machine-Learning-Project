import "./globals.css"

export const metadata = {
  title: "The Fraud Ledger · Model Brief",
  description: "IEEE-CIS Credit Card Fraud Detection · Editorial Banker Brief",
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
