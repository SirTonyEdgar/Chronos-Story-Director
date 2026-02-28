import React from 'react';
import SharedEditor from './SharedEditor';
import { Shield } from 'lucide-react';

export default function RulesTab({ profile }) {
  return (
    <SharedEditor 
      profile={profile}
      category="Rulebook" 
      icon={<Shield />} 
      color="#ef4444"
      description="📏 World Physics (Immutable): Hard laws (Magic Systems, FTL Physics) that cannot be broken."
      placeholder="e.g. Magic requires blood sacrifice, FTL travel takes 3 days to recharge..."
    />
  );
}