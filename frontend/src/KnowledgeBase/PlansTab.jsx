import React from 'react';
import SharedEditor from './SharedEditor';
import { Map } from 'lucide-react';

export default function PlansTab({ profile }) {
  return (
    <SharedEditor 
      profile={profile}
      category="Plan" 
      icon={<Map />} 
      color="#f59e0b"
      description="🗺️ Future Roadmap (Context): Plot points and schemes that haven't happened yet."
      placeholder="e.g. Operation Red Dawn, The Villain's secret master plan..."
    />
  );
}