const signupForm = document.getElementById('signupForm');
const signupError = document.getElementById('signupError');
const signupSuccess = document.getElementById('signupSuccess');
const googleSignupBtn = document.getElementById('googleSignupBtn');

const clearSignupMessages = () => {
    if (signupError) {
        signupError.textContent = '';
        signupError.classList.remove('show');
    }
    if (signupSuccess) {
        signupSuccess.textContent = '';
        signupSuccess.classList.remove('show');
    }
};

const showSignupMessage = (element, message) => {
    if (!element) return;
    element.textContent = message;
    element.classList.add('show');
};

if (signupForm) {
    signupForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        clearSignupMessages();

        const formData = new FormData(signupForm);
        const payload = {
            first_name: formData.get('first_name'),
            last_name: formData.get('last_name'),
            work_email: formData.get('work_email'),
            institution: formData.get('institution'),
            password: formData.get('password'),
            confirm_password: formData.get('confirm_password'),
        };

        if (payload.password !== payload.confirm_password) {
            showSignupMessage(signupError, 'Passwords do not match. Please try again.');
            return;
        }

        const submitButton = signupForm.querySelector('button[type="submit"]');
        if (submitButton) {
            submitButton.disabled = true;
            submitButton.textContent = 'Creating Account...';
        }

        try {
            const response = await fetch('/api/signup', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    first_name: payload.first_name,
                    last_name: payload.last_name,
                    work_email: payload.work_email,
                    institution: payload.institution,
                    password: payload.password,
                }),
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                showSignupMessage(signupError, data.detail || 'Unable to create account. Please check your details and try again.');
                return;
            }

            showSignupMessage(signupSuccess, 'Account created successfully! Redirecting to login...');
            signupForm.reset();

            setTimeout(() => {
                window.location.href = '/login';
            }, 1500);
        } catch (error) {
            console.error('Signup error:', error);
            showSignupMessage(signupError, 'An unexpected error occurred. Please try again.');
        } finally {
            if (submitButton) {
                submitButton.disabled = false;
                submitButton.textContent = 'Create Account';
            }
        }
    });
}

if (googleSignupBtn) {
    googleSignupBtn.addEventListener('click', () => {
        window.location.href = '/auth/signup/google';
    });
}
