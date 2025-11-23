// Helper function to get authorization headers
const getAuthHeaders = () => {
    const token = localStorage.getItem('access_token');
    const headers = {};
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }
    return headers;
};

const profileDetails = document.getElementById('profileDetails');

const loadProfile = async () => {
    if (!profileDetails) return;

    try {
        const response = await fetch('/api/profile', {
            headers: getAuthHeaders()
        });

        if (!response.ok) {
            if (response.status === 401) {
                // Not authenticated, redirect to login
                window.location.href = '/login';
                return;
            }
            throw new Error('Failed to load profile');
        }

        const data = await response.json();
        
        profileDetails.innerHTML = `
            <div class="profile-item">
                <span class="profile-label">Email:</span>
                <span class="profile-value">${data.work_email || 'N/A'}</span>
            </div>
            <div class="profile-item">
                <span class="profile-label">First Name:</span>
                <span class="profile-value">${data.first_name || 'N/A'}</span>
            </div>
            <div class="profile-item">
                <span class="profile-label">Last Name:</span>
                <span class="profile-value">${data.last_name || 'N/A'}</span>
            </div>
            <div class="profile-item">
                <span class="profile-label">Institution:</span>
                <span class="profile-value">${data.institution || 'N/A'}</span>
            </div>
            <div class="profile-item">
                <span class="profile-label">Auth Provider:</span>
                <span class="profile-value">${data.auth_provider || 'Email'}</span>
            </div>
        `;
    } catch (error) {
        console.error('Error loading profile:', error);
        profileDetails.innerHTML = `
            <div class="alert alert-error">Failed to load profile information. Please try again later.</div>
        `;
    }
};

// Load profile when page loads
loadProfile();

