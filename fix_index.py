import re

with open('index.html', 'r') as f:
    content = f.read()

snippet = '''<script>
(function(){
  function hideByText(text) {
    document.querySelectorAll('body *').forEach(function(el){
      if (el.children.length === 0 && el.textContent.trim() === text) {
        el.style.display = 'none';
      }
    });
  }
  window.addEventListener('DOMContentLoaded', function(){
    hideByText('accurate transcript (local whisper, free)');
    document.querySelectorAll('button').forEach(function(b){
      if (b.textContent.trim() === 'Get accurate transcript') {
        b.style.display = 'none';
      }
    });
  });
})();
</script>
</body>'''

if '</body>' in content:
    content = content.replace('</body>', snippet, 1)
    with open('index.html', 'w') as f:
        f.write(content)
    print("SUCCESS: snippet inserted before </body>")
else:
    print("ERROR: could not find </body> in index.html")
