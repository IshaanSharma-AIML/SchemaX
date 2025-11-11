// src/middleware.js

import { NextResponse } from 'next/server';

export function middleware(request) {
    // 1. Get the auth token cookie
    const token = request.cookies.get('authToken')?.value;

    // 2. Get the path the user is trying to access
    const { pathname } = request.nextUrl;

    // 3. Define protected paths and public paths
    const isProtectedRoute = pathname.startsWith('/user');
    const isPublicAuthPage = pathname === '/login' || pathname === '/register';

    // LOGIC 1: Protect '/user' routes
    // If the user has no token and is trying to access a protected route, redirect to login
    if (!token && isProtectedRoute) {
        return NextResponse.redirect(new URL('/login', request.url));
    }

    // LOGIC 2: Redirect logged-in users from auth pages
    // If the user HAS a token and is trying to access the login/register page, redirect to dashboard
    if (token && isPublicAuthPage) {
        return NextResponse.redirect(new URL('/user/dashboard', request.url));
    }

    // If neither of the above conditions are met, let the user proceed
    return NextResponse.next();
}

// This config tells the middleware which paths to run on.
// This is more efficient than running on every single request.
export const config = {
    matcher: [
        '/user/:path*',
        '/dashboard/:path*', // Also protect the dashboard
        '/login',
        '/register'
    ],
};