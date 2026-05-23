import React from 'react';
import SharedEditor from './SharedEditor';
import { Users } from 'lucide-react';

export default function CharactersTab({ profile }) {
  return (
    <SharedEditor 
      profile={profile}
      category="Character" 
      icon={<Users />} 
      color="#f97316"
      description="🧑 Character Profiles: Appearance, personality, voice, backstory, and current status. Reflect current narrative state only — future information belongs in Spoilers."
      placeholder="e.g. Sarah Chen, mid-30s, sharp-tongued corporate lawyer with a habit of disappearing for weeks at a time..."
    />
  );
}