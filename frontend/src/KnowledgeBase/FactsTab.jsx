import React from 'react';
import SharedEditor from './SharedEditor';
import { FileText } from 'lucide-react';

export default function FactsTab({ profile }) {
  return (
    <SharedEditor 
      profile={profile}
      category="Fact" 
      icon={<FileText />} 
      color="#10b981"
      description="📌 Established Truths (Current): Immediate narrative facts established in recent scenes."
      placeholder="e.g. The base was destroyed in Ch 3, The Protagonist lost their sword..."
    />
  );
}