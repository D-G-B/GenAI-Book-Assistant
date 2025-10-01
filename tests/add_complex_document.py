"""
Test utility: Add a complex document to demonstrate RAG differences
Run from project root: .venv/bin/python3 tests/add_complex_document.py
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal, LoreDocument

complex_content = """
# The Chronicles of Aethermoor: A Complete History

## Part I: The Age of Dragons (Years 0-1000)

In the beginning, there was only the Void and the First Flame. From the collision of these primordial forces emerged the Great Dragons, beings of such immense power that their very breath could shape reality itself. The five Elder Dragons - Pyraxis the Eternal Flame, Aquaria the Deep Current, Terrazon the Mountain Heart, Ventus the Storm Rider, and Chronos the Timeless - established the foundations of what would become known as Aethermoor.

The dragons did not create the world through conscious effort, but rather through their mere existence. Where Pyraxis flew, volcanoes erupted and brought forth the first lands. Aquaria's tears filled the valleys, creating the vast oceans. Terrazon's footsteps caused mountains to rise from the plains. Ventus's wings stirred the first winds that would carry seeds across the barren landscape. And Chronos, the most mysterious of all, wove the fabric of time itself, ensuring that cause would follow effect and that the universe would have a narrative structure.

For a thousand years, the dragons roamed the world alone, shaping it according to their whims. They created the seven Crystal Towers not as structures, but as living entities - massive crystalline beings that could channel and amplify magical energy across vast distances. Each tower resonated with one of the seven schools of magic: Pyromancy, Hydromancy, Terramancy, Aeromancy, Luxmancy, Umbramancy, and Chronomancy.

## Part II: The Awakening of Consciousness (Years 1000-2000)

The turning point in Aethermoor's history came when the dragons, in their curiosity, began to wonder if they were alone. Chronos, who could see through time as easily as others see through glass, perceived countless possible futures. In many of these futures, the dragons remained alone forever. But in some - a precious few - other forms of consciousness emerged.

Intrigued by this possibility, the five Elder Dragons convened for the first time in the heart of what would later become Luminspire. They deliberated for seven days and seven nights, their discussion so intense that reality itself warped around them, creating a permanent distortion in space-time that scholars would later call the Temporal Nexus.

The dragons reached a consensus: they would seed the world with consciousness. But not directly - for beings created by gods inevitably become mere puppets. Instead, they would create the conditions for consciousness to evolve naturally. They infused the world with what they called the Anima Primordialis - the First Soul - a diffuse magical essence that could, under the right circumstances, coalesce into sentient life.

## Part III: The Academy and the Seven Schools (Years 3000-4000)

The formal establishment of the Academy of Mystical Arts in the year 3127 represented the culmination of systematic magical study. Built atop the seven Crystal Towers in Luminspire, the Academy became the center of magical learning.

The Academy's curriculum was divided into seven schools:

**Pyromancy** (The School of Fire): Students learned to manipulate thermal energy, create and control flames, and understand the transformative properties of combustion. Advanced pyromancers could create flames that burned without consuming. The school's motto was "From Destruction Comes Creation." Master Pyraxis himself occasionally taught advanced courses.

**Hydromancy** (The School of Water): The study of water in all its forms. Students learned fluid dynamics, pressure manipulation, and the art of healing through water's restorative properties. Their motto: "Flow Like Water, Persistent As Tide." Aquaria served as the patron dragon of this school.

**Terramancy** (The School of Earth): Earth magic focused on stability, growth, and the manipulation of solid matter. Terramancers could sense vibrations through the ground for miles, cause earthquakes, or accelerate plant growth. Motto: "Firm Foundation, Endless Growth." Terrazon was the patron dragon.

**Aeromancy** (The School of Air): The most elusive of the elemental schools, dealing with wind, weather, and sound. Master aeromancers could fly without wings or create zones of perfect silence. Motto: "Everywhere and Nowhere." Ventus patronized this school.

**Luxmancy** (The School of Light): Light magic was about revelation and truth. Luxmancers could create illusions, see in complete darkness, or read the "light echo" of past events. Motto: "Illuminate the Unknown."

**Umbramancy** (The School of Shadow): The study of absence and concealment. Umbramancers manipulated shadows and could phase partially out of reality. Shadow was not the opposite of light, but its complement. Motto: "In Shadow, Find Shelter."

**Chronomancy** (The School of Time): The most dangerous school, dealing with time manipulation. Students learned to slow or speed time, or perceive multiple timelines. Presided over by Chronos himself. Motto: "All Moments Are Now."

## Part IV: The Great Schism (Years 4000-4500)

Around the year 4237, a fundamental disagreement emerged. The Preservationists, led by Archmage Elara Silverthread, believed magic should be used cautiously, in harmony with natural forces. The Progressives, led by chronomancer Malakai Void-Walker, argued magic's purpose was to transcend natural limitations.

The conflict centered on Malakai's proposed Ascension Ritual - a working that would elevate human consciousness to dragon level. Elara argued this would inevitably lead to catastrophe. The Dragon Council was divided: Pyraxis and Terrazon sided with Preservationists, while Ventus and Aquaria sympathized with Progressives. Chronos remained neutral.

Unable to reach consensus, the Academy split. Progressives established Voidspire Academy in the far north. For two centuries, the schools existed in cold competition.

## Part V: The Cataclysm (Year 4573)

Malakai attempted his Ascension Ritual, requiring simultaneous sacrifice of energy from all seven schools through the Crystal Towers. The ritual didn't elevate consciousness - it shattered reality. The Temporal Nexus ruptured. Time became fluid, flowing in multiple directions. Causality broke down.

Only Chronos's intervention prevented total collapse. He wove time back together over what observers experienced as seven days but what he experienced as seven thousand years. When finished, Chronos retreated to the Temporal Nexus and hasn't been seen since. Malakai and his followers became unstuck in time, flickering across multiple eras.

The Cataclysm forced academy reunion. Both sides realized neither approach alone was sufficient. The reunified Academy adopted a new philosophy: Balanced Progress.

## Part VI: The Modern Era (Years 5000-Present)

Five hundred years after the Cataclysm, Aethermoor has entered a new age. The Academy trains mages with deeper understanding of magical responsibility. The curriculum includes magical ethics, catastrophe history, and wisdom development.

The Crystal Towers now have Dampening Protocols - magical failsafes preventing another catastrophe. Relations between conscious species have evolved. Stone Singers serve as master builders. Sylphs work as messengers. The Whispering Forest trees are consulted on historical matters.

The Elder Dragons (except Chronos) maintain distance, appearing at crucial moments. New hybrid disciplines have emerged: Steam Magic (fire and water), Lightning Magic (fire and air), and controversial Void Magic (shadow and time).

Scholars detect a "Second Awakening" - new forms of consciousness emerging in the deepest oceans, beings made of pure magic rather than matter.

As current Archmage Elena Starweaver says: "We stand on the shoulders of giants and dragons alike. Our task is not to repeat their mistakes, but to honor their legacy by forging new paths forward - paths that are both bold and wise."
"""

def main():
    print("=" * 60)
    print("Adding Complex Document for RAG Testing")
    print("=" * 60)

    db = SessionLocal()

    try:
        # Check if document already exists
        existing = db.query(LoreDocument).filter(
            LoreDocument.title == "Chronicles of Aethermoor"
        ).first()

        if existing:
            print(f"\nDocument already exists (ID: {existing.id})")
            print("Updating content...")
            existing.content = complex_content
            db.commit()
            print(f"✅ Updated document ID: {existing.id}")
        else:
            # Create new document
            doc = LoreDocument(
                title="Chronicles of Aethermoor",
                filename="aethermoor_chronicles.txt",
                content=complex_content,
                source_type="text"
            )

            db.add(doc)
            db.commit()
            db.refresh(doc)

            print(f"\n✅ Created complex document (ID: {doc.id})")

        print(f"\nDocument statistics:")
        print(f"  - Characters: {len(complex_content):,}")
        print(f"  - Words: {len(complex_content.split()):,}")
        print(f"  - Lines: {len(complex_content.splitlines()):,}")

        print("\n" + "=" * 60)
        print("✅ Done! Restart your server to process this document.")
        print("=" * 60)
        print("\nSuggested test questions:")
        print("  1. What caused the Great Schism?")
        print("  2. Explain the seven schools of magic")
        print("  3. What was the Cataclysm and how was it resolved?")
        print("  4. Who are the five Elder Dragons?")
        print("  5. What is the difference between Luxmancy and Umbramancy?")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    main()
