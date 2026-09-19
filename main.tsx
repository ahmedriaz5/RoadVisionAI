import React from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

function App(){
  const [stats,setStats]=React.useState<any>({});
  React.useEffect(()=>{
    const f=async()=>setStats(await fetch("http://127.0.0.1:8000/api/stats").then(r=>r.json()));
    f(); const id=setInterval(f,1000); return()=>clearInterval(id);
  },[]);
  return <div className="app">
    <header><b>ROADVISION AI</b><span>REAL-TIME TRAFFIC INTELLIGENCE</span></header>
    <section className="grid">
      {["active_tracks","cars","motorcycles","buses","trucks","fps"].map(k=>
        <div className="card" key={k}><small>{k.replace("_"," ").toUpperCase()}</small><strong>{stats[k]??0}</strong></div>
      )}
    </section>
    <main><img src="http://127.0.0.1:8000/video"/></main>
  </div>
}
createRoot(document.getElementById("root")!).render(<App/>);
