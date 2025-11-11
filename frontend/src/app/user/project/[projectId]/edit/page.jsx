'use client'; 
// /src/app/project/[projectId]/edit/page.jsx
import dynamic from 'next/dynamic';

// Dynamically import the form component with SSR turned off.
const EditProjectForm = dynamic(
  () => import('@/components/users/project/EditProjectForm'),
  { 
    ssr: false,
    loading: () => (
        <div className="w-full max-w-7xl mx-auto p-4 lg:p-6 text-black dark:text-white text-center">
            Loading Form...
        </div>
    )
  }
);

export default function EditProjectPage() {
  return <EditProjectForm />;
}