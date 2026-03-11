import { StrictMode, Component } from "react"
import { createRoot } from "react-dom/client"
import App from "../app.jsx"

class ErrorBoundary extends Component {
  constructor(props) { super(props); this.state = { error: null } }
  static getDerivedStateFromError(e) { return { error: e } }
  render() {
    if (this.state.error) {
      return (
        <div style={{ padding: 40, fontFamily: "monospace", background: "#1a1a1a", color: "#ff6b6b", minHeight: "100vh" }}>
          <h2 style={{ color: "#ff6b6b" }}>Runtime Error</h2>
          <pre style={{ whiteSpace: "pre-wrap", marginTop: 16, color: "#fff" }}>
            {this.state.error.toString()}
            {"\n\n"}
            {this.state.error.stack}
          </pre>
        </div>
      )
    }
    return this.props.children
  }
}

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>
)
