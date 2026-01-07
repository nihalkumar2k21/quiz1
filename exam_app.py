import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import time
import hashlib

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="Quiz 1",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==================== ANTI-CHEAT JAVASCRIPT ====================
def get_anti_cheat_js(roll_number="", time_elapsed=""):
    """Enhanced anti-cheat JavaScript with watermark showing elapsed time"""
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

    /* Watermark - shows Roll and Time Elapsed */
    body::before {{
        content: "Roll: {roll_number} | {time_elapsed}";
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
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0

# ==================== HELPER FUNCTIONS ====================
def roll_to_seed(roll_number: str) -> int:
    """Convert any roll number string into a stable numeric seed."""
    # Add salt to ensure different questions than previous exam
    salted = roll_number + "_v2_jan2026"
    return int(hashlib.md5(salted.encode()).hexdigest(), 16) % (2**32)

def generate_questions(roll_number):
    """Generate NEW questions - completely different from previous exam"""
    np.random.seed(roll_to_seed(roll_number))
    
    questions = []
    
    # NEW Q1: Multi-layer Perceptron Forward Pass
    w1 = np.random.uniform(0.2, 0.8, 2).round(2)
    w2 = np.random.uniform(0.3, 0.9, 2).round(2)
    b1 = round(np.random.uniform(-0.3, 0.3), 2)
    b2 = round(np.random.uniform(-0.4, 0.4), 2)
    x1, x2 = np.random.randint(1, 4, 2)
    
    # Calculate answer
    h1 = max(0, w1[0] * x1 + w1[1] * x2 + b1)  # ReLU
    output = w2[0] * h1 + w2[1] * 1 + b2  # Linear (second input is bias=1)
    
    questions.append({
        'id': 'q1',
        'marks': 2,
        'title': 'Multi-Layer Perceptron Forward Pass',
        'question_text': f"""A 2-layer neural network with:
- **Layer 1 weights:** w1 = {list(w1)}, bias b1 = {b1}
- **Layer 2 weights:** w2 = {list(w2)}, bias b2 = {b2}
- **Input:** x = [{x1}, {x2}]
- **Activation:** ReLU for hidden layer, Linear for output

*ReLU(z) = max(0, z)*""",
        'parts': [
            {'id': 'a', 'text': 'Calculate hidden layer output: h1 = ReLU(w1[0]×x[0] + w1[1]×x[1] + b1)', 'marks': 1},
            {'id': 'b', 'text': 'Calculate final output: y = w2[0]×h1 + w2[1]×1 + b2', 'marks': 1}
        ],
        'answer_key': {
            'h1': round(h1, 4),
            'output': round(output, 4)
        }
    })
    
    # NEW Q2: Batch vs Stochastic Gradient Descent
    batch_size = int(np.random.choice([4, 8, 16]))
    dataset_size = int(np.random.choice([100, 200, 500]))
    epochs = int(np.random.choice([5, 10, 20]))
    
    # Calculate answer
    sgd_updates = dataset_size * epochs
    batch_updates = (dataset_size // batch_size) * epochs
    
    questions.append({
        'id': 'q2',
        'marks': 2,
        'title': 'Gradient Descent Variants',
        'question_text': f"""You have a dataset with **{dataset_size} samples**. You train for **{epochs} epochs**.

**Batch Gradient Descent:** Uses all samples per update  
**Stochastic Gradient Descent (SGD):** Uses 1 sample per update  
**Mini-batch GD:** Uses {batch_size} samples per update""",
        'parts': [
            {'id': 'a', 'text': f'How many **weight updates** occur with **SGD** over {epochs} epochs?', 'marks': 1},
            {'id': 'b', 'text': f'How many **weight updates** occur with **mini-batch GD (batch size={batch_size})** over {epochs} epochs?', 'marks': 1}
        ],
        'answer_key': {
            'sgd_updates': sgd_updates,
            'minibatch_updates': batch_updates
        }
    })
    
    # NEW Q3: Loss Function and Gradient
    pred = round(np.random.uniform(2.0, 4.0), 2)
    target = round(np.random.uniform(5.0, 7.0), 2)
    
    # MSE Loss and gradient
    loss = (pred - target) ** 2
    gradient = 2 * (pred - target)
    
    questions.append({
        'id': 'q3',
        'marks': 2,
        'title': 'Loss Function & Gradient Calculation',
        'question_text': f"""You're using **Mean Squared Error (MSE)** loss: L = (ŷ - y)²

Current prediction: **ŷ = {pred}**  
Target value: **y = {target}**

*Derivative: ∂L/∂ŷ = 2(ŷ - y)*""",
        'parts': [
            {'id': 'a', 'text': 'Calculate the **loss value** L', 'marks': 1},
            {'id': 'b', 'text': 'Calculate the **gradient** ∂L/∂ŷ', 'marks': 1}
        ],
        'answer_key': {
            'loss': round(loss, 4),
            'gradient': round(gradient, 4)
        }
    })
    
    # NEW Q4: Activation Function Properties
    val1 = round(np.random.uniform(-2.0, -0.5), 2)
    val2 = round(np.random.uniform(1.0, 3.0), 2)
    
    # ReLU outputs
    relu_val1 = max(0, val1)
    relu_val2 = max(0, val2)
    
    # Tanh approximation
    tanh_val2 = round(np.tanh(val2), 4)
    
    questions.append({
        'id': 'q4',
        'marks': 2,
        'title': 'Activation Functions: ReLU vs Tanh',
        'question_text': f"""Given two values: **z1 = {val1}** and **z2 = {val2}**

**ReLU:** f(z) = max(0, z)  
**Tanh:** f(z) = (e^z - e^(-z))/(e^z + e^(-z))""",
        'parts': [
            {'id': 'a', 'text': f'Calculate **ReLU(z1)** and **ReLU(z2)**', 'marks': 1},
            {'id': 'b', 'text': f'Which activation function can output **negative values**: ReLU or Tanh? Justify with an example.', 'marks': 1}
        ],
        'answer_key': {
            'relu_z1': relu_val1,
            'relu_z2': relu_val2,
            'negative_capable': 'Tanh (can output values from -1 to 1)'
        }
    })
    
    # NEW Q5: Neural Network Architecture - Parameters Count
    input_features = int(np.random.choice([20, 50, 100]))
    hidden_neurons = int(np.random.choice([32, 64, 128]))
    output_classes = int(np.random.choice([3, 5, 10]))
    
    # Calculate parameters
    params_layer1 = (input_features * hidden_neurons) + hidden_neurons
    params_layer2 = (hidden_neurons * output_classes) + output_classes
    total_params = params_layer1 + params_layer2
    
    questions.append({
        'id': 'q5',
        'marks': 2,
        'title': 'Neural Network Parameters Calculation',
        'question_text': f"""A neural network architecture:
- **Input layer:** {input_features} features
- **Hidden layer:** {hidden_neurons} neurons
- **Output layer:** {output_classes} classes (softmax)

Each layer has **bias terms**.""",
        'parts': [
            {'id': 'a', 'text': f'Calculate **total parameters** (weights + biases) between input and hidden layer', 'marks': 1},
            {'id': 'b', 'text': f'Calculate **total parameters** in the entire network (all layers)', 'marks': 1}
        ],
        'answer_key': {
            'params_layer1': params_layer1,
            'total_params': total_params
        }
    })
    
    return questions

def display_timer(start_time):
    """Display countdown timer with progress bar and return elapsed time string"""
    exam_duration = 20 * 60  # 20 minutes in seconds
    elapsed = (datetime.now() - start_time).total_seconds()
    remaining = max(0, exam_duration - elapsed)
    
    minutes = int(remaining // 60)
    seconds = int(remaining % 60)
    
    # Calculate elapsed time for watermark
    elapsed_minutes = int(elapsed // 60)
    elapsed_seconds = int(elapsed % 60)
    elapsed_str = f"{elapsed_minutes:02d}:{elapsed_seconds:02d} elapsed"
    
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
    
    return remaining > 0, elapsed_str

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
    
    # Add all answers (now includes parts a, b, c)
    for q in st.session_state.questions:
        for part in q['parts']:
            answer_key = f"{q['id']}_part_{part['id']}"
            response_data[answer_key] = st.session_state.answers.get(answer_key, '')
        # Store answer key for grading reference
        response_data[f"{q['id']}_answer_key"] = str(q['answer_key'])
    
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
    # Calculate elapsed time for watermark
    elapsed_time_str = ""
    if st.session_state.started and st.session_state.start_time:
        elapsed = (datetime.now() - st.session_state.start_time).total_seconds()
        elapsed_min = int(elapsed // 60)
        elapsed_sec = int(elapsed % 60)
        elapsed_time_str = f"{elapsed_min:02d}:{elapsed_sec:02d} elapsed"
    
    # Inject anti-cheat JavaScript
    roll = st.session_state.roll_number if st.session_state.roll_number else ""
    st.markdown(get_anti_cheat_js(roll, elapsed_time_str), unsafe_allow_html=True)
    
    st.title("🧠 Deep Learning Quiz 1 - Retake")
    st.markdown("---")
    
    # ==================== LOGIN SCREEN ====================
    if not st.session_state.started:
        st.markdown("### 🎓 Student Login")
        
        # Warning banner
        st.markdown("""
        <div style='background-color: #ff4444; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
            <h3 style='color: white; margin: 0;'>⚠️ EXAM INTEGRITY NOTICE</h3>
            <p style='color: white; margin: 10px 0 0 0; font-size: 14px;'>
                • All activities are monitored and logged<br>
                • Violations will result in mark deductions or disciplinary action<br>
                • This exam must be completed within 20 minutes<br>
                • <strong>NEW QUESTIONS - Different from previous attempt</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.warning("📋 **Instructions:**")
        st.markdown("""
        - This is a **20-minute** timed exam worth **10 marks** (5 questions × 2 marks each)
        - Questions appear **one at a time** - navigate using Next/Previous buttons
        - Each question has **multiple parts (a, b)** with separate answer boxes
        - Each student has **different numerical parameters** in questions
        - **DO NOT** refresh the page or close the browser tab
        - You must **show your step-by-step calculations** for each part
        - Click **"Start Exam"** only when you are completely ready
        - **Timer will NOT auto-update** - click Next/Previous to refresh
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
                st.session_state.current_question = 0
                st.rerun()
            else:
                st.error("❌ Please enter a valid roll number!")
    
    # ==================== EXAM SCREEN ====================
    else:
        # Check if time is up
        time_remaining, elapsed_str = display_timer(st.session_state.start_time)
        
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
        
        # Get current question
        current_q = st.session_state.questions[st.session_state.current_question]
        total_questions = len(st.session_state.questions)
        
        # Question progress indicator
        st.markdown(f"### Question {st.session_state.current_question + 1} of {total_questions}")
        st.progress((st.session_state.current_question + 1) / total_questions)
        
        # Question title and text
        st.markdown(f"#### {current_q['title']} ({current_q['marks']} marks)")
        st.info(current_q['question_text'])
        
        st.markdown("---")
        
        # Display parts with separate text areas
        st.markdown("### 📝 Your Answers:")
        for part in current_q['parts']:
            st.markdown(f"**Part ({part['id']})** *[{part['marks']} mark{'s' if part['marks'] != 1 else ''}]*: {part['text']}")
            
            answer_key = f"{current_q['id']}_part_{part['id']}"
            st.session_state.answers[answer_key] = st.text_area(
                f"Your answer for part ({part['id']}):",
                value=st.session_state.answers.get(answer_key, ''),
                height=150,
                key=answer_key,
                placeholder="Write your step-by-step solution here...",
                label_visibility="collapsed"
            )
            st.markdown("")
        
        st.markdown("---")
        
        # Navigation buttons
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            if st.session_state.current_question > 0:
                if st.button("⬅️ Previous", use_container_width=True):
                    st.session_state.current_question -= 1
                    st.rerun()
        
        with col2:
            # Show which questions have been answered
            answered_count = 0
            for q in st.session_state.questions:
                for part in q['parts']:
                    if st.session_state.answers.get(f"{q['id']}_part_{part['id']}", '').strip():
                        answered_count += 1
            
            total_parts = sum(len(q['parts']) for q in st.session_state.questions)
            st.info(f"📊 {answered_count}/{total_parts} parts")
        
        with col3:
            if st.session_state.current_question < total_questions - 1:
                if st.button("Next ➡️", use_container_width=True):
                    st.session_state.current_question += 1
                    st.rerun()
        
        with col4:
            if st.button("✅ Submit", type="primary", use_container_width=True):
                # Check if answers are provided
                answered_parts = sum(
                    1 for q in st.session_state.questions 
                    for part in q['parts']
                    if st.session_state.answers.get(f"{q['id']}_part_{part['id']}", '').strip()
                )
                
                if answered_parts == 0:
                    st.error("❌ You haven't answered any parts! Please provide your answers.")
                else:
                    save_responses()
                    st.success(f"""
                    ✅ **Exam Submitted Successfully!**
                    
                    **Roll Number:** {st.session_state.roll_number}  
                    **Parts Answered:** {answered_parts}/{total_parts}  
                    **Submission Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
                    **Tab Switches:** {st.session_state.tab_switches}  
                    **Screenshot Attempts:** {st.session_state.screenshot_attempts}
                    
                    Your responses have been recorded. You may now close this window.
                    """)
                    st.balloons()
                    st.stop()

# ==================== RUN APPLICATION ====================
if __name__ == "__main__":
    main()
