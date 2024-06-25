### Algorithm Description for Extracting LaTeX Equations with Context

This algorithm is designed to extract LaTeX equations from a given text, along with their surrounding context. The algorithm is specifically tailored to identify equations in three formats commonly used in LaTeX documents: inline equations delimited by single dollar signs (`$...$`), block equations delimited by double dollar signs (`$$...$$`), and equations within `\begin{equation}...\end{equation}` environments. Below, we provide a detailed explanation of the algorithm, suitable for inclusion in a technical paper for the AAAI conference.

#### Algorithm Outline

1. **Initialization**: 
   - The algorithm initializes an empty list `equations_with_context` to store the extracted equations along with their context.
   - It sets a pointer `i` to zero, which will be used to traverse the text character by character.

2. **Context Cleaning Function**:
   - The `clean_context` function is defined to remove non-English words and special characters from the context. This function uses a regular expression to retain only letters, digits, and basic punctuation, ensuring that the context is clean and relevant.
   - The context is further processed to keep only English words, providing a cleaner and more focused context.

3. **Text Traversal**:
   - The algorithm enters a loop that continues until the end of the text is reached.
   - Within the loop, the current position and a 100-character window before it (`context_before`) are tracked and cleaned using the `clean_context` function.

4. **Equation Detection**:
   - The algorithm checks for the presence of equations starting with `$$`, `$`, or `\begin{equation}` at the current position.
     - For `$$` (block equations), it finds the closing `$$` and extracts the equation along with a 100-character window after it (`context_after`).
     - For `$` (inline equations), it finds the closing `$` and extracts the equation along with a 100-character window after it.
     - For `\begin{equation}`, it finds the corresponding `\end{equation}` and extracts the equation along with a 100-character window after it.
   - If none of these patterns are found, the pointer `i` is incremented to move to the next character.

5. **Context Limitation**:
   - Both `context_before` and `context_after` are limited to the last and first 10 words, respectively, ensuring that the context is concise and relevant.
   - The algorithm also ensures that `context_after` does not inadvertently include another equation by splitting at the next `$` or `\begin{equation}` if found.

6. **Equation and Context Storage**:
   - If the extracted equation is 100 characters or fewer, it, along with the cleaned and limited contexts, is added to the `equations_with_context` list.

7. **Return Result**:
   - Once the entire text has been processed, the algorithm returns the `equations_with_context` list, containing dictionaries with the keys `context_before`, `equation`, and `context_after`.

#### Example

Given the text:
```latex
Here is some text with an equation $E = mc^2$ in it.
Here is another equation $$a^2 + b^2 = c^2$$ which is between double dollar signs.
Also, consider the following equation environment:
\begin{equation}
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
\end{equation}
```

The algorithm will identify and extract the following equations and contexts:
```python
[
    {
        'context_before': 'Here is some text with an equation',
        'equation': '$E = mc^2$',
        'context_after': 'in it.'
    },
    {
        'context_before': 'Here is another equation',
        'equation': '$$a^2 + b^2 = c^2$$',
        'context_after': 'which is between double'
    },
    {
        'context_before': 'Also, consider the following equation environment:',
        'equation': '\\begin{equation}\nx = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}\n\\end{equation}',
        'context_after': ''
    }
]
```

#### Conclusion

This algorithm is efficient in extracting LaTeX equations from text while providing relevant context. It ensures that the context is clean and concise, making it suitable for various natural language processing tasks, including context-aware equation analysis and retrieval. By using a combination of regular expressions and manual text traversal, the algorithm achieves a balance between speed and accuracy, making it a robust choice for processing LaTeX documents in academic and research settings.