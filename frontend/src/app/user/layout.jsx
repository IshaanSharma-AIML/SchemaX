// app/user/layout.jsx
import Header from '@/components/users/layouts/Header';

export const metadata = {
    title: 'SchemaX Dashboard',
    description: 'Your AI-powered database command center',
};

export default function UserPagesLayout({ children }) {
    return (
        <div className="h-screen w-screen bg-gray-50 dark:bg-gray-900 overflow-hidden flex flex-col">
            <Header />
            <main className="flex-1 overflow-hidden">
                <div className="w-full h-full">
                    {children}
                </div>
            </main>
        </div>
    );
}
