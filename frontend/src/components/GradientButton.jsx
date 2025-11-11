import Link from 'next/link';

const GradientButton = ({ href, children, className = '' }) => (
    <Link href={href}>
        <span className={`inline-block bg-gradient-to-r from-blue-500 to-blue-700 hover:from-blue-600 hover:to-blue-800 text-white font-semibold rounded-lg transition-all duration-200 glow-button ${className}`}>
            {children}
        </span>
    </Link>
);

export default GradientButton;