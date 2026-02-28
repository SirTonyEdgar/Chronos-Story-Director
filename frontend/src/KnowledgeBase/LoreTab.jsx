import React from 'react';
import SharedEditor from './SharedEditor';
import { Book } from 'lucide-react';

export default function LoreTab({ profile }) {
  return (
    <SharedEditor 
      profile={profile}
      category="Lore" 
      icon={<Book />} 
      color="#3b82f6"
      description="📖 Story Bible (Background): Permanent history, geography, and character backstories."
      placeholder="e.g. The Fall of the Old Empire, The nature of mana crystals..."
    />
  );
}