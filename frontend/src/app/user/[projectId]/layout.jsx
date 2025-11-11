import Sidebar from '@/components/Sidebar';

export const metadata = {
    title: 'User Panel',
    description: 'Your AI-powered business command center',
};

export default function UserPagesLayout({ children }) {
    return (
        <div className="w-full h-full flex">
            <Sidebar />
            <div className="flex-1 h-full overflow-hidden">
                {children}
            </div>
        </div>
    );
}