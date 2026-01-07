import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import time
import hashlib

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="Quiz 1",
    #page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==================== ANTI-CHEAT JAVASCRIPT ====================
def get_anti_cheat_js(roll_number="", timestamp=""):
    """Enhanced anti-cheat JavaScript with watermark"""
    return f"""
    <script>
    // Disable right-click
    document.addEventListener('contextmenu', event => {{
        event.preventDefault();
        alert('⚠️ Right-click is disabled during the exam!');
        return false;
    }});

    // Disable keyboard shortcuts
    document.addEventListener('keydown', function(e) {{
        // F12, Ctrl+Shift+I, Ctrl+Shift+J, Ctrl+U, Ctrl+S
        if (e.keyCode == 123 || 
            (e.ctrlKey && e.shiftKey && (e.keyCode == 73 || e.keyCode == 74)) ||
            (e.ctrlKey && e.keyCode == 85) ||
            (e.ctrlKey && e.keyCode == 83)) {{
            e.preventDefault();
            alert('⚠️ Developer tools are disabled!');
            return false;
        }}
        
        // Ctrl+C, Ctrl+X (copy/cut)
        if (e.ctrlKey && (e.keyCode == 67 || e.keyCode == 88)) {{
            e.preventDefault();
            alert('⚠️ Copy/Cut is disabled during the exam!');
            return false;
        }}
        
        // Ctrl+P (print)
        if (e.ctrlKey && e.keyCode == 80) {{
            e.preventDefault();
            alert('⚠️ Printing is disabled!');
            return false;
        }}
    }});

    // Detect Print Screen
    document.addEventListener('keyup', function(e) {{
        if (e.keyCode == 44 || e.key == 'PrintScreen') {{
            alert('⚠️ SCREENSHOT DETECTED!\\n\\nThis attempt has been logged and timestamped.\\nYour instructor will be notified.');
            window.parent.postMessage({{type: 'SCREENSHOT_ATTEMPT'}}, '*');
        }}
    }});

    // Disable text selection on questions (allow in textarea)
    document.addEventListener('selectstart', function(e) {{
        if (!e.target.matches('textarea') && !e.target.matches('input')) {{
            e.preventDefault();
            return false;
        }}
    }});

    // Detect tab switching
    document.addEventListener('visibilitychange', function() {{
        if (document.hidden) {{
            alert('⚠️ TAB SWITCH DETECTED!\\n\\nSwitching tabs during the exam is prohibited.\\nThis violation has been logged.');
            window.parent.postMessage({{type: 'TAB_SWITCH'}}, '*');
        }}
    }});

    // Disable drag-and-drop
    document.addEventListener('dragstart', function(e) {{
        e.preventDefault();
        return false;
    }});

    // Warn on page unload
    window.addEventListener('beforeunload', function(e) {{
        e.preventDefault();
        e.returnValue = 'Your exam is in progress! Are you sure you want to leave?';
        return e.returnValue;
    }});

    // Disable paste
    document.addEventListener('paste', function(e) {{
        if (!e.target.matches('textarea') && !e.target.matches('input')) {{
            e.preventDefault();
            alert('⚠️ Pasting is disabled!');
            return false;
        }}
    }});

    // Monitor focus loss
    var focusLossCount = 0;
    window.addEventListener('blur', function() {{
        focusLossCount++;
        if (focusLossCount > 2) {{
            alert('⚠️ Please keep focus on the exam window!');
        }}
    }});
    </script>

    <style>
    /* Disable text selection globally */
    body {{
        user-select: none;
        -webkit-user-select: none;
        -moz-user-select: none;
        -ms-user-select: none;
    }}

    /* Allow selection in input fields */
    textarea, input {{
        user-select: text !important;
        -webkit-user-select: text !important;
        -moz-user-select: text !important;
        -ms-user-select: text !important;
    }}

    /* Watermark */
    body::before {{
        content: "Roll: {roll_number} | {timestamp}";
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%) rotate(-45deg);
        font-size: 60px;
        color: rgba(255, 0, 0, 0.08);
        z-index: 9999;
        pointer-events: none;
        white-space: nowrap;
        font-weight: bold;
    }}

    /* Disable print */
    @media print {{
        body {{
            display: none;
        }}
    }}

    /* Hide Streamlit elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    </style>
    """

# ==================== SESSION STATE INITIALIZATION ====================
if 'started' not in st.session_state:
    st.session_state.started = False
if 'roll_number' not in st.session_state:
    st.session_state.roll_number = None
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'answers' not in st.session_state:
    st.session_state.answers = {}
if 'tab_switches' not in st.session_state:
    st.session_state.tab_switches = 0
if 'screenshot_attempts' not in st.session_state:
    st.session_state.screenshot_attempts = 0
if 'questions' not in st.session_state:
    st.session_state.questions = []

# ==================== HELPER FUNCTIONS ====================
def roll_to_seed(roll_number: str) -> int:
    """Convert any roll number string into a stable numeric seed."""
    return int(hashlib.md5(roll_number.encode()).hexdigest(), 16) % (2**32)

def generate_questions(roll_number):
    """Generate unique questions based on roll number"""
    np.random.seed(roll_to_seed(roll_number))
    
    questions = []
    
    # Q1: Perceptron calculation
    weights = np.random.uniform(-1, 1, 3).round(2)
    bias = round(np.random.uniform(-0.5, 0.5), 2)
    inputs = np.random.randint(1, 5, 3)
    
    net_input = round(np.dot(weights, inputs) + bias, 4)
    sigmoid_output = round(1 / (1 + np.exp(-net_input)), 4)
    
    questions.append({
        'id': 'q1',
        'marks': 2,
        'question': f"""**Question 1 (2 marks):** 
A perceptron has weights **w = {list(weights)}** and bias **b = {bias}**.  
For input **x = {list(inputs)}**, calculate:

(a) Net input  
(b) Output after sigmoid activation σ(z) = 1/(1+e^(-z))

**Show your calculations step by step.**""",
        'answer_key': {
            'net_input': net_input,
            'sigmoid_output': sigmoid_output
        }
    })
    
    # Q2: Gradient Descent
    lr = round(np.random.choice([0.001, 0.01, 0.1]), 3)
    current_weight = 0.5
    gradient_pos = 2.0
    gradient_neg = -2.0
    
    updated_pos = round(current_weight - lr * gradient_pos, 4)
    updated_neg = round(current_weight - lr * gradient_neg, 4)
    
    questions.append({
        'id': 'q2',
        'marks': 2,
        'question': f"""**Question 2 (2 marks):**
You're using gradient descent with learning rate **α = {lr}**.  
Current weight **w = {current_weight}**, gradient **∂L/∂w = {gradient_pos}**.

(a) What will be the updated weight after one gradient descent step?  
(b) If the gradient was {gradient_neg} instead, what would be the new weight?

**Show your calculations.**""",
        'answer_key': {
            'updated_weight_positive': updated_pos,
            'updated_weight_negative': updated_neg
        }
    })
    
    # Q3: Backpropagation
    w1 = round(np.random.uniform(0.3, 0.7), 2)
    w2 = round(np.random.uniform(0.4, 0.8), 2)
    x_input = 2.0
    target = 5.0
    
    h = max(0, w1 * x_input)  # ReLU
    y = w2 * h  # Linear output
    loss = (y - target) ** 2
    
    questions.append({
        'id': 'q3',
        'marks': 2,
        'question': f"""**Question 3 (2 marks):**
A 2-layer network: Input(x) → Hidden(h) → Output(y)

- **w1 = {w1}** (input to hidden weight)
- **w2 = {w2}** (hidden to output weight)  
- **Activation:** ReLU for hidden layer, Linear for output layer
- **Input:** x = {x_input}, **Target:** t = {target}
- **Loss:** L = (y - t)²

Calculate:  
(a) Forward pass: h = ? and y = ?  
(b) Loss value L = ?

**Show all steps.**""",
        'answer_key': {
            'h': round(h, 4),
            'y': round(y, 4),
            'loss': round(loss, 4)
        }
    })
    
    # Q4: Activation Functions
    threshold = round(np.random.choice([0.4, 0.5, 0.6]), 1)
    outputs = [0.8, 0.3, 0.9, 0.1]
    step_outputs = [1 if x > threshold else 0 for x in outputs]
    sigmoid_03 = round(1 / (1 + np.exp(-0.3)), 4)
    
    questions.append({
        'id': 'q4',
        'marks': 2,
        'question': f"""**Question 4 (2 marks):**
You have outputs **[0.8, 0.3, 0.9, 0.1]** from a binary classifier.

(a) Using **step activation** with threshold = **{threshold}**, what are the final outputs?  
(b) If you use **sigmoid** instead, will the output for 0.3 be **> {threshold}** or **< {threshold}**? Calculate sigmoid(0.3) and explain.

**Show your reasoning.**""",
        'answer_key': {
            'step_outputs': step_outputs,
            'sigmoid_0.3': sigmoid_03
        }
    })
    
    # Q5: MLP Architecture
    img_size = int(np.random.choice([16, 28, 32]))
    hidden1 = int(np.random.choice([64, 128, 256]))
    num_classes = 10
    
    input_size = img_size * img_size
    params = (input_size * hidden1) + hidden1  # weights + biases
    
    questions.append({
        'id': 'q5',
        'marks': 2,
        'question': f"""**Question 5 (2 marks):**
Design a 3-layer MLP for **{img_size}×{img_size}** grayscale images → 10 classes.  
Hidden layer 1 has **{hidden1}** neurons.

Calculate:  
(a) **Input layer size** = ?  
(b) **Number of parameters** (weights + biases) between input and hidden layer 1 = ?  
(c) Which **activation function** should be used for the output layer and **why**?

**Show calculations for (a) and (b).**""",
        'answer_key': {
            'input_size': input_size,
            'parameters': params,
            'output_activation': 'Softmax (for multiclass classification)'
        }
    })
    
    return questions

def display_timer(start_time):
    """Display countdown timer with progress bar"""
    exam_duration = 20 * 60  # 20 minutes in seconds
    elapsed = (datetime.now() - start_time).total_seconds()
    remaining = max(0, exam_duration - elapsed)
    
    minutes = int(remaining // 60)
    seconds = int(remaining % 60)
    
    # Color coding based on time remaining
    if remaining <= 60:  # Last 1 minute
        st.markdown(f"### ⏰ Time Remaining: <span style='color:red; font-size:28px;'>{minutes:02d}:{seconds:02d}</span>", 
                   unsafe_allow_html=True)
    elif remaining <= 300:  # Last 5 minutes
        st.markdown(f"### ⏰ Time Remaining: <span style='color:orange;'>{minutes:02d}:{seconds:02d}</span>", 
                   unsafe_allow_html=True)
    else:
        st.markdown(f"### ⏰ Time Remaining: {minutes:02d}:{seconds:02d}")
    
    # Progress bar
    progress = 1 - (remaining / exam_duration)
    st.progress(progress)
    
    return remaining > 0

def save_responses():
    """Save student responses to CSV with all metadata"""
    response_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'roll_number': st.session_state.roll_number,
        'start_time': st.session_state.start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'duration_seconds': int((datetime.now() - st.session_state.start_time).total_seconds()),
        'tab_switches': st.session_state.tab_switches,
        'screenshot_attempts': st.session_state.screenshot_attempts,
    }
    
    # Add all answers
    for q in st.session_state.questions:
        response_data[f"{q['id']}_answer"] = st.session_state.answers.get(q['id'], '')
        # Store answer key for grading reference
        response_data[f"{q['id']}_key"] = str(q['answer_key'])
    
    df = pd.DataFrame([response_data])
    
    # Append to CSV (create if doesn't exist)
    try:
        existing_df = pd.read_csv('exam_responses.csv')
        df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        pass
    
    df.to_csv('exam_responses.csv', index=False)
    
    # Also save individual response
    df.to_csv(f'response_{st.session_state.roll_number}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', index=False)

# ==================== MAIN APPLICATION ====================
def main():
    # Inject anti-cheat JavaScript
    current_time = datetime.now().strftime('%H:%M:%S')
    roll = st.session_state.roll_number if st.session_state.roll_number else ""
    st.markdown(get_anti_cheat_js(roll, current_time), unsafe_allow_html=True)
    
    st.title("🧠 Deep Learning Exam")
    st.markdown("---")
    
    # ==================== LOGIN SCREEN ====================
    if not st.session_state.started:
        st.markdown("### 🎓 Student Login")
        
        # Warning banner
        st.markdown("""
        <div style='background-color: #ff4444; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
            <h3 style='color: white; margin: 0;'>⚠️ EXAM INTEGRITY NOTICE</h3>
            <p style='color: white; margin: 10px 0 0 0; font-size: 14px;'>
                • All activities are monitored and logged.<br>
                • Violations will result in mark deductions or disciplinary action<br>
                • This exam must be completed within 20 minutes
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.warning("📋 **Instructions:**")
        st.markdown("""
        - This is a **20-minute** timed exam worth **10 marks** (5 questions × 2 marks each)
        - Each student has **different numerical parameters** in questions
        - **DO NOT** refresh the page or close the browser tab
        - Right-click, copy-paste, and F12 developer tools are **disabled**
        - Tab switching and screenshot attempts are **logged**
        - You must **show your calculations** for all numerical questions
        - Click **"Start Exam"** only when you are completely ready
        - Once started, the timer **cannot be paused**
        """)
        
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            roll_number = st.text_input(
                "🎫 Enter your Roll Number:",
                max_chars=20,
                placeholder="e.g., 2021/CS/001"
            )
        
        st.markdown("")
        
        if st.button("🚀 Start Exam", type="primary", use_container_width=True):
            if roll_number and roll_number.strip():
                st.session_state.roll_number = roll_number.strip().upper()
                st.session_state.start_time = datetime.now()
                st.session_state.started = True
                st.session_state.questions = generate_questions(roll_number.strip())
                st.rerun()
            else:
                st.error("❌ Please enter a valid roll number!")
    
    # ==================== EXAM SCREEN ====================
    else:
        # Check if time is up
        time_remaining = display_timer(st.session_state.start_time)
        
        if not time_remaining:
            st.error("⏰ **TIME'S UP!** Your exam has been automatically submitted.")
            save_responses()
            st.balloons()
            st.success(f"""
            ✅ **Exam Submitted Successfully!**
            
            **Roll Number:** {st.session_state.roll_number}  
            **Submission Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Your responses have been recorded. You may now close this window.
            """)
            st.stop()
        
        # Student info and violations
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**👤 Roll:** {st.session_state.roll_number}")
        with col2:
            if st.session_state.tab_switches > 0:
                st.warning(f"⚠️ Tab switches: {st.session_state.tab_switches}")
        with col3:
            if st.session_state.screenshot_attempts > 0:
                st.error(f"📸 Screenshots: {st.session_state.screenshot_attempts}")
        
        st.markdown("---")
        
        # Display all questions
        total_marks = sum(q['marks'] for q in st.session_state.questions)
        st.markdown(f"### 📝 Answer ALL Questions (Total: {total_marks} marks)")
        
        for i, q in enumerate(st.session_state.questions, 1):
            st.markdown(f"#### Question {i}")
            st.markdown(q['question'])
            
            st.session_state.answers[q['id']] = st.text_area(
                f"✍️ Your Answer for Question {i}:",
                value=st.session_state.answers.get(q['id'], ''),
                height=180,
                key=f"answer_{q['id']}",
                placeholder="Write your answer here. Show all calculation steps clearly..."
            )
            st.markdown("---")
        
        # Submission section
        st.markdown("### 📤 Exam Submission")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("✅ Make sure you've answered all questions before submitting!")
        with col2:
            if st.button("✅ Submit Exam", type="primary", use_container_width=True):
                # Check if at least some answers are provided
                answered = sum(1 for ans in st.session_state.answers.values() if ans.strip())
                
                if answered == 0:
                    st.error("❌ You haven't answered any questions! Please provide your answers.")
                else:
                    save_responses()
                    st.success(f"""
                    ✅ **Exam Submitted Successfully!**
                    
                    **Roll Number:** {st.session_state.roll_number}  
                    **Questions Attempted:** {answered}/{len(st.session_state.questions)}  
                    **Submission Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
                    **Tab Switches:** {st.session_state.tab_switches}  
                    **Screenshot Attempts:** {st.session_state.screenshot_attempts}
                    
                    Your responses have been recorded. You may now close this window.
                    """)
                    st.balloons()
                    st.stop()
        
        # Auto-refresh every second for timer update
        time.sleep(1)
        st.rerun()

# ==================== RUN APPLICATION ====================
if __name__ == "__main__":
    main()
