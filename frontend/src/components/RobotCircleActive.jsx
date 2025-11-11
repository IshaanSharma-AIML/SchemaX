import { FaRobot } from 'react-icons/fa';
const RobotCircleActive = () => {
    return (
        <div className="flex justify-center mb-4">
            <div className="relative">
                <div className="w-20 h-20 rounded-full bg-gray-800 flex items-center justify-center mb-6 border-2 border-gray-700">
                    <FaRobot className="text-sky-400 text-4xl" />
                </div>
                <div className="absolute -top-1 -right-1 w-5 h-5 bg-teal-400 rounded-full border-2 border-gray-900 animate-pulse"></div>
            </div>
        </div>
    )
}

export default RobotCircleActive;