import "./globals.css"

export const metadata = {
  title: "Fraud Detection · Live Dashboard",
  description: "IEEE-CIS Credit Card Fraud Detection · Banker dashboard",
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
