import React from 'react';
import SharedEditor from './SharedEditor';
import { Shield } from 'lucide-react';

export default function FactionTab({ profile }) {
  return (
    <SharedEditor 
      profile={profile}
      category="Faction" 
      icon={<Shield />} 
      color="#f59e0b"
      description="⚔️ Faction Profiles: Voice, known information, blind spots, communication style, and standing biases."
      placeholder="e.g. The Night's Watch writes in terse military dispatches, never mentions wildlings without contempt..."
    />
  );
}