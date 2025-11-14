const loginForm = document.getElementById('loginForm');
const errorMessage = document.getElementById('errorMessage');
const successMessage = document.getElementById('successMessage');
const googleLoginBtn = document.getElementById('googleLoginBtn');

const clearMessages = () => {
    if (errorMessage) {
        errorMessage.textContent = '';
        errorMessage.classList.remove('show');
    }
    if (successMessage) {
        successMessage.textContent = '';
        successMessage.classList.remove('show');
    }
};

const showMessage = (el, message) => {
    if (!el) return;
    el.textContent = message;
    el.classList.add('show');
};

if (loginForm) {
    loginForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        clearMessages();

        const formData = new FormData(loginForm);
        const payload = {
            work_email: formData.get('work_email'),
            password: formData.get('password'),
        };

        const submitButton = loginForm.querySelector('button[type="submit"]');
        if (submitButton) {
            submitButton.disabled = true;
            submitButton.textContent = 'Signing In...';
        }

        try {
            const response = await fetch('/api/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload),
            });

            const data = await response.json();

            if (!response.ok) {
                showMessage(errorMessage, data.detail || 'Login failed. Please check your credentials.');
                return;
            }

            showMessage(successMessage, 'Login successful! Redirecting...');

            if (data.access_token) {
                localStorage.setItem('access_token', data.access_token);
            }

            setTimeout(() => {
                window.location.href = '/dashboard';
            }, 1200);
        } catch (error) {
            console.error('Login error:', error);
            showMessage(errorMessage, 'An unexpected error occurred. Please try again.');
        } finally {
            if (submitButton) {
                submitButton.disabled = false;
                submitButton.textContent = 'Sign In';
            }
        }
    });
}

if (googleLoginBtn) {
    googleLoginBtn.addEventListener('click', () => {
        window.location.href = '/auth/login/google';
    });
}