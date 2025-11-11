// 'use client';

// import { FaCheck, FaTimes, FaRocket, FaCrown, FaBuilding, FaArrowRight, FaShieldAlt, FaAward } from 'react-icons/fa';
// import GradientButton from '@/components/GradientButton';

// export default function PricingPage() {
//     const plans = [
//         {
//             name: "Starter",
//             price: "Free",
//             period: "forever",
//             description: "Perfect for individuals and small teams getting started",
//             features: [
//                 "Up to 3 database connections",
//                 "100 queries per month",
//                 "Basic AI assistance",
//                 "Email support",
//                 "Standard response time"
//             ],
//             limitations: [
//                 "Limited to 1 project",
//                 "No advanced analytics",
//                 "Basic data visualization"
//             ],
//             buttonText: "Get Started Free",
//             buttonLink: "/register",
//             popular: false,
//             icon: <FaRocket className="text-2xl" />
//         },
//         {
//             name: "Professional",
//             price: "$29",
//             period: "per month",
//             description: "Ideal for growing businesses and data teams",
//             features: [
//                 "Unlimited database connections",
//                 "1,000 queries per month",
//                 "Advanced AI assistance",
//                 "Priority support",
//                 "Custom dashboards",
//                 "Data visualization",
//                 "Export capabilities",
//                 "Team collaboration"
//             ],
//             limitations: [],
//             buttonText: "Start Free Trial",
//             buttonLink: "/register",
//             popular: true,
//             icon: <FaCrown className="text-2xl" />
//         },
//         {
//             name: "Enterprise",
//             price: "Custom",
//             period: "pricing",
//             description: "For large organizations with advanced needs",
//             features: [
//                 "Everything in Professional",
//                 "Unlimited queries",
//                 "Dedicated support",
//                 "Custom integrations",
//                 "Advanced security",
//                 "SLA guarantee",
//                 "On-premise deployment",
//                 "Custom training"
//             ],
//             limitations: [],
//             buttonText: "Contact Sales",
//             buttonLink: "/contact",
//             popular: false,
//             icon: <FaBuilding className="text-2xl" />
//         }
//     ];

//     return (
//         <div className="h-full bg-white dark:bg-gray-900 overflow-hidden">
//             <div className="h-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 overflow-y-auto">
//                 {/* Header */}
//                 <div className="text-center mb-16">
//                     <h1 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-4">
//                         SchemaX Pricing
//                     </h1>
//                     <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
//                         Choose the plan that fits your needs. Start free and upgrade as you grow.
//                     </p>
                    
//                 </div>

//                 {/* Pricing Cards */}
//                 <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
//                     {plans.map((plan, index) => (
//                         <div
//                             key={index}
//                             className={`relative bg-white dark:bg-gray-800 rounded-2xl p-8 border-2 transition-all duration-300 hover:shadow-xl ${
//                                 plan.popular
//                                     ? 'border-blue-500 shadow-lg scale-105'
//                                     : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
//                             }`}
//                         >
//                             {plan.popular && (
//                                 <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
//                                     <span className="bg-blue-500 text-white px-4 py-1 rounded-full text-sm font-medium">
//                                         Most Popular
//                                     </span>
//                                 </div>
//                             )}

//                             <div className="text-center mb-8">
//                                 <div className="flex justify-center mb-4">
//                                     <div className={`w-16 h-16 rounded-full flex items-center justify-center ${
//                                         plan.popular 
//                                             ? 'bg-blue-100 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400'
//                                             : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
//                                     }`}>
//                                         {plan.icon}
//                                     </div>
//                                 </div>
//                                 <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
//                                     {plan.name}
//                                 </h3>
//                                 <div className="mb-4">
//                                     <span className="text-4xl font-bold text-gray-900 dark:text-white">
//                                         {plan.price}
//                                     </span>
//                                     <span className="text-gray-600 dark:text-gray-400 ml-2">
//                                         {plan.period}
//                                     </span>
//                                 </div>
//                                 <p className="text-gray-600 dark:text-gray-300">
//                                     {plan.description}
//                                 </p>
//                             </div>

//                             <div className="space-y-4 mb-8">
//                                 {plan.features.map((feature, featureIndex) => (
//                                     <div key={featureIndex} className="flex items-center">
//                                         <FaCheck className="text-green-500 mr-3 flex-shrink-0" />
//                                         <span className="text-gray-700 dark:text-gray-300">{feature}</span>
//                                     </div>
//                                 ))}
//                                 {plan.limitations.map((limitation, limitationIndex) => (
//                                     <div key={limitationIndex} className="flex items-center">
//                                         <FaTimes className="text-red-500 mr-3 flex-shrink-0" />
//                                         <span className="text-gray-500 dark:text-gray-400">{limitation}</span>
//                                     </div>
//                                 ))}
//                             </div>

//                             <GradientButton
//                                 href={plan.buttonLink}
//                                 className={`w-full py-3 px-6 rounded-lg font-semibold transition-all duration-200 ${
//                                     plan.popular
//                                         ? 'bg-gradient-to-r from-blue-500 to-blue-700 hover:from-blue-600 hover:to-blue-800 text-white'
//                                         : 'bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-white hover:bg-gray-200 dark:hover:bg-gray-600'
//                                 }`}
//                             >
//                                 <span className="flex items-center justify-center gap-2">
//                                     {plan.buttonText}
//                                     <FaArrowRight className="w-4 h-4" />
//                                 </span>
//                             </GradientButton>
//                         </div>
//                     ))}
//                 </div>

//             </div>
//         </div>
//     );
// }
