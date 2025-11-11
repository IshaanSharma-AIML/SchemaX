// src/components/users/modals/DeleteConfirmationModal.jsx
'use client';

import { AlertDialog, AlertDialogTrigger, AlertDialogContent, AlertDialogHeader, AlertDialogFooter, AlertDialogTitle, AlertDialogDescription, AlertDialogAction, AlertDialogCancel } from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { FaExclamationTriangle } from 'react-icons/fa';

const DeleteConfirmationModal = ({ isOpen, onClose, onConfirm, title, message, isLoading }) => {
    return (
        <AlertDialog open={isOpen} onOpenChange={open => { if (!open) onClose(); }}>
            <AlertDialogContent>
                <AlertDialogHeader>
                    <div className="flex items-center gap-3">
                        <span className="flex items-center justify-center h-10 w-10 rounded-full bg-red-500/10">
                        <FaExclamationTriangle className="h-6 w-6 text-red-400" aria-hidden="true" />
                        </span>
                        <AlertDialogTitle>{title || 'Delete Item'}</AlertDialogTitle>
                    </div>
                    <AlertDialogDescription className="text-gray-500 dark:text-gray-400">
                                {message || 'Are you sure? This action cannot be undone.'}
                    </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                    <AlertDialogCancel asChild>
                        <Button variant="secondary" disabled={isLoading} onClick={onClose}>Cancel</Button>
                    </AlertDialogCancel>
                    <AlertDialogAction asChild>
                        <Button variant="destructive" disabled={isLoading} onClick={onConfirm}>
                         {isLoading ? (
                                <span className="animate-spin rounded-full h-5 w-5 border-b-2 border-white inline-block"></span>
                         ) : 'Delete'}
                        </Button>
                    </AlertDialogAction>
                </AlertDialogFooter>
            </AlertDialogContent>
        </AlertDialog>
    );
};

export default DeleteConfirmationModal;