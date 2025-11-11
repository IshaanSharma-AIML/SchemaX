// src/app/layout.jsx

import { Poppins } from "next/font/google";
import "./globals.css";
import StoreProvider from "@/lib/store/StoreProvider";
import { Toaster } from "react-hot-toast"; // Import the Toaster component
import { ThemeProvider } from '@/contexts/ThemeContext'; // Import ThemeProvider
// Configure the Poppins font
const poppins = Poppins({
  subsets: ["latin"],
  weight: ['400', '500', '600', '700'], // Specify weights you want to use
  variable: "--font-poppins", // Create a CSS variable
  display: 'swap',
});

export const metadata = {
  title: "AI Business Agent",
  description: "Your 24/7 AI-powered business consultant",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={`${poppins.variable} w-full h-full`}>
      {/* Apply the font variable to the body */}
      <body className={`antialiased bg-white dark:bg-[#0f172a] w-full h-full m-0 p-0`}>
        <ThemeProvider>
          <StoreProvider>
            {/* Toaster component for beautiful, non-blocking notifications */}
            <Toaster
              position="top-center" // You can change the position
              reverseOrder={false}
              toastOptions={{
                // Define default options
                className: '',
                duration: 5000,
                style: {
                  background: '#363636',
                  color: '#fff',
                },
                // Default options for specific types
                success: {
                  duration: 3000,
                  theme: {
                    primary: 'green',
                    secondary: 'black',
                  },
                },
              }}
            />
            {children}
          </StoreProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}